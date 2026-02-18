import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { ImgedaStack } from '../lib/imgeda-stack';

// DockerImageCode.fromImageAsset requires Docker to build the image.
// Mock it for unit tests so tests run without Docker.
jest.mock('aws-cdk-lib/aws-ecr-assets', () => {
  const original = jest.requireActual('aws-cdk-lib/aws-ecr-assets');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { Construct } = require('constructs');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const iam = require('aws-cdk-lib/aws-iam');
  return {
    ...original,
    DockerImageAsset: class MockDockerImageAsset extends Construct {
      public readonly imageUri: string;
      public readonly assetHash: string;
      public readonly repository: any;
      constructor(scope: any, id: string) {
        super(scope, id);
        this.imageUri = '123456789012.dkr.ecr.us-east-1.amazonaws.com/mock:latest';
        this.assetHash = 'mockhash';
        this.repository = {
          repositoryArn: 'arn:aws:ecr:us-east-1:123456789012:repository/mock',
          repositoryName: 'mock',
          grantPull: (grantee: any) => iam.Grant.drop(grantee, 'mock'),
          grantPullPush: (grantee: any) => iam.Grant.drop(grantee, 'mock'),
        };
      }
      // eslint-disable-next-line @typescript-eslint/no-empty-function
      addResourceMetadata() {}
    },
  };
});

describe('ImgedaStack', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const stack = new ImgedaStack(app, 'TestStack');
    template = Template.fromStack(stack);
  });

  test('creates input and output S3 buckets', () => {
    template.resourceCountIs('AWS::S3::Bucket', 2);
  });

  test('output bucket has lifecycle rule for partials', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      LifecycleConfiguration: {
        Rules: [
          {
            Prefix: 'partials/',
            ExpirationInDays: 7,
            Status: 'Enabled',
          },
        ],
      },
    });
  });

  test('creates 5 Lambda functions', () => {
    template.resourceCountIs('AWS::Lambda::Function', 5);
  });

  test('creates a Step Functions state machine', () => {
    template.resourceCountIs('AWS::StepFunctions::StateMachine', 1);
  });

  test('creates CloudWatch alarms for Lambda errors', () => {
    template.resourceCountIs('AWS::CloudWatch::Alarm', 5);
  });

  test('buckets block public access', () => {
    template.hasResourceProperties('AWS::S3::Bucket', {
      PublicAccessBlockConfiguration: {
        BlockPublicAcls: true,
        BlockPublicPolicy: true,
        IgnorePublicAcls: true,
        RestrictPublicBuckets: true,
      },
    });
  });

  test('stack has expected outputs', () => {
    template.hasOutput('InputBucketName', {});
    template.hasOutput('OutputBucketName', {});
    template.hasOutput('StateMachineArn', {});
  });

  test('buckets default to RETAIN removal policy', () => {
    template.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Retain',
    });
  });
});

describe('ImgedaStack with autoCleanup', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const stack = new ImgedaStack(app, 'TestCleanupStack', {
      autoCleanup: true,
    });
    template = Template.fromStack(stack);
  });

  test('buckets use DESTROY removal policy', () => {
    template.hasResource('AWS::S3::Bucket', {
      DeletionPolicy: 'Delete',
    });
  });

  test('adds auto-delete custom resource Lambda', () => {
    // autoDeleteObjects adds a custom resource handler (5 app + 1 auto-delete = 6)
    template.resourceCountIs('AWS::Lambda::Function', 6);
  });
});
