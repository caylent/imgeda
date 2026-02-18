import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { ImgedaStack } from '../lib/imgeda-stack';

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

  test('Lambda functions use Python 3.12 runtime', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      Runtime: 'python3.12',
    });
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

  test('creates a Lambda layer', () => {
    template.resourceCountIs('AWS::Lambda::LayerVersion', 1);
  });

  test('stack has expected outputs', () => {
    template.hasOutput('InputBucketName', {});
    template.hasOutput('OutputBucketName', {});
    template.hasOutput('StateMachineArn', {});
  });
});
