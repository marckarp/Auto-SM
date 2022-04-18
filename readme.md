## Auto SageMaker

## Setup

1. Clone GitHub repo
2. `python setup.py bdist_wheel`
3. `pip install dist/automodel-0.0.1-py3-none-any.whl`
4. `automodel-configure --module-name automodel.configure`

The last command will create a file called `config.ini` in $HOME to store your AWS role. 

## Requirements

```
[AWS]
role = arn:aws:iam::<ACCOUNT>:role/service-role/AmazonSageMaker-ExecutionRole-20210412T095523
```

## Folder Structure for saved model

Models must be saved in a folder with the following structure - 

```
model/
    model.pkl|.pth|.joblib|.pt
    code/ 
        inference.py
        requirements.txt
        files_needed_by_inference.py
```

