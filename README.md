# Snowflake as Data Source for training an ML Model with Amazon SageMaker

This repository provides an example for how to use Snowflake as a source of training data for training a machine learning model in Amazon SageMaker. The training data is downloaded directly from a Snowflake table into a training instance rather than being first downloaded into an S3 bucket. A custom container is created for training the ML model, it is based on the [SageMaker XGBoost container image](https://github.com/aws/sagemaker-xgboost-container) and the [snowflake-python connector](https://pypi.org/project/snowflake-connector-python/) is packaged into this container.

The training dataset used in the sample notebook is the [California Housing Dataset](https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html) that is available for download from the Internet using the `sklearn.datasets`.

## Installation

The notebook in this repository works well with SageMaker Notebooks. It **does not** work with SageMaker Studio Notebooks because it has code for creating a `docker` container and SageMaker Studio Notebook kernels do not currently have `docker` installed.

The following steps describe the steps to be following _before_ running the notebook code.

1. Create a free account with Snowflake and upload the California housing dataset. Detailed instructions are available in [`snowflake-instructions`](./snowflake-instructions.md) file.

1. Create an IAM role with the necessary permissions. This can be done by running the following commands in `CloudShell`.

   ```{{bash}}
   role=SagemakerSnowflakeExampleRole
   aws iam create-role --role-name ${role} --assume-role-policy-document file://iam/trust_role_policy_document.json
   account_id=$(aws sts get-caller-identity | jq -r '.Account')
   sed -i "s/__ACCOUNT_ID__/${account_id}/g" iam/ECR_permissions.json
   aws iam put-role-policy --role-name ${role} --policy-name ECR_permissions --policy-document file://iam/ECR_permissions.json
   ```

1. Create a SageMaker Notebook with all default options selected except for the IAM role, select the ``SagemakerSnowflakeExampleRole` created in the previous step.

1. Clone the repository in the SageMaker Notebook.

## Usage

1. Clone this repository in the SageMaker Notebook created via the steps described above.

1. Run the code in the [`sagemaker-snowflake-example`](./sagemaker-snowflake-example.ipynb) notebook.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. See [CONTRIBUTING](./CONTRIBUTING.md)

## Roadmap

- [ ] Code cleanup

See the open issues for a full list of proposed features (and known issues).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
