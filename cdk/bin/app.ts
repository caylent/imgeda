#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { ImgedaStack } from '../lib/imgeda-stack';

const app = new cdk.App();

// Use -c autoCleanup=true for test deployments that should auto-delete on destroy
const autoCleanup = app.node.tryGetContext('autoCleanup') === 'true';

new ImgedaStack(app, 'ImgedaStack', {
  description: 'Serverless image dataset EDA pipeline using Step Functions',
  autoCleanup,
});
