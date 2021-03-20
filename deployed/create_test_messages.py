import boto3

# Create SQS client
sqs = boto3.resource('sqs', region_name='us-east-1')

queue = sqs.get_queue_by_name(QueueName='mlgovhawk-test')
#response = queue.send_message(MessageBody='boto3', MessageAttributes={
#    'Author': {
#        'StringValue': 'Daniel',
#        'DataType': 'String'
#    }
#})

#Send message to SQS queue
response = queue.send_message(
    MessageAttributes={
        'bill_version_id': {
            'DataType': 'Number',
            'StringValue': '1234'
        },
        'bill_id': {
            'DataType': 'Number',
            'StringValue': '5678'
        },
        'revision_number': {
            'DataType': 'Number',
            'StringValue': '6'
        },
        'partisan_lean': {
            'DataType': 'Number',
            'StringValue': '6'
        },
        'session_id': {
            'DataType': 'Number',
            'StringValue': '1'
        },
        'chamber_id': {
            'DataType': 'Number',
            'StringValue': '2'
        }

    },
    MessageBody=(
        'abc'
    )
)

print(response['MessageId'])