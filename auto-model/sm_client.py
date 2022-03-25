import boto3
import sagemaker

class AutoSMClient():
    """ """
    def __init__(self, **kwargs) -> None:
        ''' '''
        boto_session = boto3.session.Session()
        self._region_ = boto_session.region_name
        sagemaker_session = sagemaker.Session()
        self._s3_client_ = boto_session.resource('s3')
        self._sm_client_ = boto3.client("sagemaker")
        
        self._default_bucket_ = sagemaker_session.default_bucket()
        self._role_ = kwargs['role']

    @property
    def AutoS3Client(self):
        ''' '''
        return self._s3_client_.meta.client
    
    @property
    def AutoSagemakerClient(self):
        ''' '''
        return self._sm_client_

    @property
    def Region(self):
        ''' '''
        return self._region_

    @property
    def Role(self):
        ''' '''
        return self._role_

    @property
    def DefaultBucket(self):
        ''' '''
        return self._default_bucket_

if __name__ == '__main__':
    ''' '''
    auto_sm_client = AutoSMClient(role = 'arn:aws:iam::682101512330:role/service-role/AmazonSageMaker-ExecutionRole-20210412T095523')
    sm_client = auto_sm_client.AutoSagemakerClient
    s3_client = auto_sm_client.AutoS3Client
    