import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import { Construct } from 'constructs';
import * as path from 'path';

export interface ImgedaStackProps extends cdk.StackProps {
  /** Auto-delete bucket contents on stack destroy. Use for dev/test only. */
  readonly autoCleanup?: boolean;
}

export class ImgedaStack extends cdk.Stack {
  public readonly inputBucket: s3.Bucket;
  public readonly outputBucket: s3.Bucket;
  public readonly stateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props?: ImgedaStackProps) {
    super(scope, id, props);

    const autoCleanup = props?.autoCleanup ?? false;
    const removalPolicy = autoCleanup
      ? cdk.RemovalPolicy.DESTROY
      : cdk.RemovalPolicy.RETAIN;

    // --- S3 Buckets ---
    this.inputBucket = new s3.Bucket(this, 'InputBucket', {
      removalPolicy,
      autoDeleteObjects: autoCleanup,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
    });

    this.outputBucket = new s3.Bucket(this, 'OutputBucket', {
      removalPolicy,
      autoDeleteObjects: autoCleanup,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      lifecycleRules: [
        {
          prefix: 'partials/',
          expiration: cdk.Duration.days(7),
        },
      ],
    });

    // --- Docker Image Lambda ---
    // All 5 Lambda functions share one Docker image built from lambda/Dockerfile.
    // Each function sets an ACTION env var to route to the correct handler.
    const projectRoot = path.resolve(__dirname, '../..');
    const imageCode = lambda.DockerImageCode.fromImageAsset(projectRoot, {
      file: 'lambda/Dockerfile',
      exclude: [
        'cdk',
        'tests',
        '.venv',
        '.git',
        'node_modules',
        '.github',
        '.claude',
        'dist',
        'build',
        '*.egg-info',
        '.ruff_cache',
        '.mypy_cache',
        '.pytest_cache',
        'htmlcov',
      ],
    });

    const listImagesFn = new lambda.DockerImageFunction(this, 'ListImagesFn', {
      code: imageCode,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(5),
      description: 'List images in S3 bucket and split into batches',
      environment: { ACTION: 'list_images' },
    });

    const analyzeBatchFn = new lambda.DockerImageFunction(this, 'AnalyzeBatchFn', {
      code: imageCode,
      memorySize: 2048,
      timeout: cdk.Duration.minutes(10),
      ephemeralStorageSize: cdk.Size.gibibytes(2),
      description: 'Analyze a batch of images',
      environment: { ACTION: 'analyze_batch' },
    });

    const mergeManifestsFn = new lambda.DockerImageFunction(this, 'MergeManifestsFn', {
      code: imageCode,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(10),
      description: 'Merge partial manifests into final JSONL',
      environment: { ACTION: 'merge_manifests' },
    });

    const aggregateFn = new lambda.DockerImageFunction(this, 'AggregateFn', {
      code: imageCode,
      memorySize: 1024,
      timeout: cdk.Duration.minutes(5),
      description: 'Compute aggregate statistics from manifest',
      environment: { ACTION: 'aggregate' },
    });

    const generatePlotsFn = new lambda.DockerImageFunction(this, 'GeneratePlotsFn', {
      code: imageCode,
      memorySize: 2048,
      timeout: cdk.Duration.minutes(10),
      description: 'Generate visualization plots from manifest',
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
    //
    // Execution input schema:
    // {
    //   "input_bucket": "<bucket-with-images>",
    //   "prefix": "images/",
    //   "output_bucket": "<bucket-for-results>"
    // }
    //
    // Each task uses `resultPath` to preserve the full state across steps,
    // and `payload` to construct the handler-specific input from state fields.

    const listImagesTask = new tasks.LambdaInvoke(this, 'ListImages', {
      lambdaFunction: listImagesFn,
      payload: sfn.TaskInput.fromObject({
        'bucket.$': '$.input_bucket',
        'prefix.$': '$.prefix',
      }),
      resultPath: '$.list_result',
      payloadResponseOnly: true,
    });

    const analyzeBatchTask = new tasks.LambdaInvoke(this, 'AnalyzeBatch', {
      lambdaFunction: analyzeBatchFn,
      payloadResponseOnly: true,
    });

    const analyzeMap = new sfn.Map(this, 'AnalyzeBatches', {
      itemsPath: '$.list_result.batches',
      maxConcurrency: 10,
      resultPath: '$.analyze_results',
      itemSelector: {
        'source_bucket.$': '$.input_bucket',
        'keys.$': '$$.Map.Item.Value',
        'output_bucket.$': '$.output_bucket',
        'output_key.$':
          "States.Format('partials/batch-{}.jsonl', $$.Map.Item.Index)",
      },
    });
    analyzeMap.itemProcessor(analyzeBatchTask);

    const mergeTask = new tasks.LambdaInvoke(this, 'MergeManifests', {
      lambdaFunction: mergeManifestsFn,
      payload: sfn.TaskInput.fromObject({
        'bucket.$': '$.output_bucket',
        'analyze_results.$': '$.analyze_results',
        'output_key': 'manifests/manifest.jsonl',
        'input_dir.$':
          "States.Format('s3://{}/{}', $.input_bucket, $.prefix)",
      }),
      resultPath: '$.merge_result',
      payloadResponseOnly: true,
    });

    const aggregateTask = new tasks.LambdaInvoke(this, 'Aggregate', {
      lambdaFunction: aggregateFn,
      payload: sfn.TaskInput.fromObject({
        'bucket.$': '$.output_bucket',
        'manifest_key.$': '$.merge_result.output_key',
        'output_key': 'summary/summary.json',
      }),
      resultPath: '$.aggregate_result',
      payloadResponseOnly: true,
    });

    const generatePlotsTask = new tasks.LambdaInvoke(this, 'GeneratePlots', {
      lambdaFunction: generatePlotsFn,
      payload: sfn.TaskInput.fromObject({
        'bucket.$': '$.output_bucket',
        'manifest_key.$': '$.merge_result.output_key',
        'output_prefix': 'plots/',
      }),
      resultPath: '$.plots_result',
      payloadResponseOnly: true,
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
