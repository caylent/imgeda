#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { ImgedaStack } from '../lib/imgeda-stack';

const app = new cdk.App();
new ImgedaStack(app, 'ImgedaStack', {
  description: 'Serverless image dataset EDA pipeline using Step Functions',
});
