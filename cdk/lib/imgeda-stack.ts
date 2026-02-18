import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import { Construct } from 'constructs';

export class ImgedaStack extends cdk.Stack {
  public readonly inputBucket: s3.Bucket;
  public readonly outputBucket: s3.Bucket;
  public readonly stateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // --- S3 Buckets ---
    this.inputBucket = new s3.Bucket(this, 'InputBucket', {
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    });

    this.outputBucket = new s3.Bucket(this, 'OutputBucket', {
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          prefix: 'partials/',
          expiration: cdk.Duration.days(7),
        },
      ],
    });

    // --- Lambda Layer ---
    const imgedaLayer = new lambda.LayerVersion(this, 'ImgedaLayer', {
      code: lambda.Code.fromAsset('../', {
        bundling: {
          image: lambda.Runtime.PYTHON_3_12.bundlingImage,
          command: [
            'bash', '-c',
            'pip install . -t /asset-output/python && cp -r src/imgeda /asset-output/python/',
          ],
        },
      }),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
      description: 'imgeda package and dependencies',
    });

    // --- Lambda Functions ---
    const commonLambdaProps: Partial<lambda.FunctionProps> = {
      runtime: lambda.Runtime.PYTHON_3_12,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(5),
      layers: [imgedaLayer],
      handler: 'imgeda.lambda_handler.handler.handler',
    };

    const listImagesFn = new lambda.Function(this, 'ListImagesFn', {
      ...commonLambdaProps,
      code: lambda.Code.fromInline('# handler in layer'),
      handler: 'imgeda.lambda_handler.handler.handler',
      description: 'List images in S3 bucket and split into batches',
      timeout: cdk.Duration.minutes(5),
      environment: { ACTION: 'list_images' },
    });

    const analyzeBatchFn = new lambda.Function(this, 'AnalyzeBatchFn', {
      ...commonLambdaProps,
      code: lambda.Code.fromInline('# handler in layer'),
      handler: 'imgeda.lambda_handler.handler.handler',
      description: 'Analyze a batch of images',
      memorySize: 2048,
      timeout: cdk.Duration.minutes(10),
      ephemeralStorageSize: cdk.Size.gibibytes(2),
      environment: { ACTION: 'analyze_batch' },
    });

    const mergeManifestsFn = new lambda.Function(this, 'MergeManifestsFn', {
      ...commonLambdaProps,
      code: lambda.Code.fromInline('# handler in layer'),
      handler: 'imgeda.lambda_handler.handler.handler',
      description: 'Merge partial manifests into final JSONL',
      timeout: cdk.Duration.minutes(10),
      environment: { ACTION: 'merge_manifests' },
    });

    const aggregateFn = new lambda.Function(this, 'AggregateFn', {
      ...commonLambdaProps,
      code: lambda.Code.fromInline('# handler in layer'),
      handler: 'imgeda.lambda_handler.handler.handler',
      description: 'Compute aggregate statistics from manifest',
      environment: { ACTION: 'aggregate' },
    });

    const generatePlotsFn = new lambda.Function(this, 'GeneratePlotsFn', {
      ...commonLambdaProps,
      code: lambda.Code.fromInline('# handler in layer'),
      handler: 'imgeda.lambda_handler.handler.handler',
      description: 'Generate visualization plots from manifest',
      memorySize: 2048,
      timeout: cdk.Duration.minutes(10),
      environment: { ACTION: 'generate_plots' },
    });

    // --- IAM: least-privilege S3 access ---
    this.inputBucket.grantRead(listImagesFn);
    this.inputBucket.grantRead(analyzeBatchFn);

    this.outputBucket.grantReadWrite(analyzeBatchFn);
    this.outputBucket.grantReadWrite(mergeManifestsFn);
    this.outputBucket.grantReadWrite(aggregateFn);
    this.outputBucket.grantReadWrite(generatePlotsFn);

    // --- Step Functions Workflow ---
    const listImagesTask = new tasks.LambdaInvoke(this, 'ListImages', {
      lambdaFunction: listImagesFn,
      outputPath: '$.Payload',
    });

    const analyzeBatchTask = new tasks.LambdaInvoke(this, 'AnalyzeBatch', {
      lambdaFunction: analyzeBatchFn,
      outputPath: '$.Payload',
    });

    const analyzeMap = new sfn.Map(this, 'AnalyzeBatches', {
      itemsPath: '$.batches',
      maxConcurrency: 10,
      resultPath: '$.analyze_results',
    });
    analyzeMap.itemProcessor(analyzeBatchTask);

    const mergeTask = new tasks.LambdaInvoke(this, 'MergeManifests', {
      lambdaFunction: mergeManifestsFn,
      outputPath: '$.Payload',
    });

    const aggregateTask = new tasks.LambdaInvoke(this, 'Aggregate', {
      lambdaFunction: aggregateFn,
      outputPath: '$.Payload',
    });

    const generatePlotsTask = new tasks.LambdaInvoke(this, 'GeneratePlots', {
      lambdaFunction: generatePlotsFn,
      outputPath: '$.Payload',
    });

    const definition = listImagesTask
      .next(analyzeMap)
      .next(mergeTask)
      .next(aggregateTask)
      .next(generatePlotsTask);

    this.stateMachine = new sfn.StateMachine(this, 'ImgedaPipeline', {
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      timeout: cdk.Duration.hours(2),
    });

    // --- CloudWatch Alarms ---
    const allFunctions = [
      listImagesFn,
      analyzeBatchFn,
      mergeManifestsFn,
      aggregateFn,
      generatePlotsFn,
    ];

    for (const fn of allFunctions) {
      new cloudwatch.Alarm(this, `${fn.node.id}ErrorAlarm`, {
        metric: fn.metricErrors({ period: cdk.Duration.minutes(5) }),
        threshold: 1,
        evaluationPeriods: 1,
        alarmDescription: `Error alarm for ${fn.functionName}`,
      });
    }

    // --- Outputs ---
    new cdk.CfnOutput(this, 'InputBucketName', {
      value: this.inputBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'OutputBucketName', {
      value: this.outputBucket.bucketName,
    });
    new cdk.CfnOutput(this, 'StateMachineArn', {
      value: this.stateMachine.stateMachineArn,
    });
  }
}
