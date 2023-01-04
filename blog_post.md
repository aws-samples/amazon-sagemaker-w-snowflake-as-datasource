# Using Snowflake as a data source to train ML models with Amazon SageMaker

_Amit Arora_, _Divya Muralidharan_

Amazon SageMaker requires that the training data for a machine learning model be present either in [S3 or in EFS or in FSX for Lustre](https://aws.amazon.com/blogs/machine-learning/choose-the-best-data-source-for-your-amazon-sagemaker-training-job/). In order to train a model using data stored outside of the three supported storage services, the data first needs to be ingested into one of these services (typically S3). This requires building a data pipeline (using tools such as [Amazon SageMaker Data Wrangler](https://aws.amazon.com/sagemaker/data-wrangler/)) to move data into S3. However, this may create a data management challenge in terms of managing the lifecycle of this data, access controls and more. In such situations it may be desirable to have the data accessible to SageMaker _without_ the intermediate storage of data in S3.

This post shows a way to do this using [Snowflake](https://www.snowflake.com/) as the data source and by downloading the data directly from Snowflake into a SageMaker Training Job instance.

## Solution overview

We use the [California Housing Dataset](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html) as a training dataset for this post and train an ML model to predict the median house value for each district. We add this data to Snowflake as a new table. We create a custom training container which downloads data directly from the Snowflake table into the training instance **_rather than first downloading the data into an S3 bucket_**. Once the data is downloaded into the training instance, the custom training script performs data preparation tasks and then trains the machine learning model using the [XGBoost Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html). All code for this blog post including the SageMaker Snowflake notebook is available in this [GitHub repo](https://github.com/aws-samples/amazon-sagemaker-w-snowflake-as-datasource).

![Architecture](img/snowflake-sagemaker-Page-1.png)

## Walkthrough

Navigate through the steps provided in this post to download data from Snowflake directly into SageMaker Training instance.

![Flowchart](img/snowflake-sagemaker-Page-2.png)

Brief intro to the walkthrough.
Sentence about the steps:
• Step
• Step
• Step
Link to CloudFormation stack or GitHub repo:
[Launch stack button]

### Prerequisites

For this walkthrough, you should have the following prerequisites:
• An AWS account
• AWS resources
• Any third-party software or hardware
• Any specialized knowledge

#### Create step section

Paragraph with link to service documentation topics for basic procedures or more information.
To take action with a procedure

1. Log in to the [ServiceX console]().
2. Take action A.
3. Take action B.
4. Run the following command:
aws ecs create-cluster --cluster-name example --region us-east-1
5. Finish.
The following screenshot shows the best practices for images:

The following screenshot shows what not to do with images:

Code sample introduction

```{bash}
#include <stdio.h>
int main()
{
   // printf() displays the string inside quotation
   printf("Hello, World!");
   return 0;
}
```

## Cleaning up

To avoid incurring future charges, delete the resources.

## Conclusion

Restate the post purpose and add next steps. The calls to action can include related content.
[Optional] Author bio
Photo Three sentences introducing the author’s AWS role, experience and interests, and a lighthearted personal note.

Suggested tags:
