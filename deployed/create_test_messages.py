import boto3

# Create SQS client
sqs = boto3.resource('sqs', region_name='us-east-1')

queue = sqs.get_queue_by_name(QueueName='mlgovhawk-test')

#queue.purge()
#time.sleep(65)

#Send message to SQS queue
response = queue.send_message(
    MessageAttributes={
        'bill_version_id': {
            'DataType': 'Number',
            'StringValue': '2740636'
        },
        'bill_id': {
            'DataType': 'Number',
            'StringValue': '1346334'
        },
        'version_number': {
            'DataType': 'Number',
            'StringValue': '1'
        },
        'partisan_lean': {
            'DataType': 'Number',
            'StringValue': '0.525'
        },
        'session_id': {
            'DataType': 'Number',
            'StringValue': '520'
        },
        'chamber_id': {
            'DataType': 'Number',
            'StringValue': '2'
        },
        'plain_url':{
            'DataType': 'String',
            'StringValue': 'https://lis.virginia.gov/cgi-bin/legp604.exe?202+ful+HB5001'
        }

    },
    MessageBody=(
        'abc'
    )
)

print(response['MessageId'])

response = queue.send_message(
    MessageAttributes={
        'bill_version_id': {
            'DataType': 'Number',
            'StringValue': '2127866'
        },
        'bill_id': {
            'DataType': 'Number',
            'StringValue': '1098149'
        },
        'version_number': {
            'DataType': 'Number',
            'StringValue': '1'
        },
        'partisan_lean': {
            'DataType': 'Number',
            'StringValue': '0.236'
        },
        'session_id': {
            'DataType': 'Number',
            'StringValue': '562'
        },
        'chamber_id': {
            'DataType': 'Number',
            'StringValue': '1'
        },
        'plain_url':{
            'DataType': 'String',
            'StringValue': 'http://www.capitol.tn.gov/Bills/111/Bill/SJR0006.pdf'
        }

    },
    MessageBody=(
        'None'
    )
)

print(response['MessageId'])