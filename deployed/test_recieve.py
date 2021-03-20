import boto3
# Get the service resource
sqs = boto3.resource('sqs', region_name='us-east-1')

# Get the queue
queue = sqs.get_queue_by_name(QueueName='mlgovhawk-test')

# Process messages by printing out body and optional author name
for message in queue.receive_messages(MessageAttributeNames=['All']):
    # Get the custom author message attribute if it was set
    revision_id = ''
    if message.message_attributes is not None:
        bill_version_id = message.message_attributes.get('bill_version_id').get('StringValue')
        bill_id = message.message_attributes.get('bill_id').get('StringValue')
        revision_number = message.message_attributes.get('revision_number').get('StringValue')
        partisan_lean = message.message_attributes.get('partisan_lean').get('StringValue')
        session_id = message.message_attributes.get('session_id').get('StringValue')
        chamber_id = message.message_attributes.get('chamber_id').get('StringValue')
    
    # Print out the body and author (if set)
    print(f'bill_version_id:, {bill_version_id}')
    print(f'bill_id:, {bill_id}')
    print(f'revision_number:, {revision_number}')
    print(f'partisan_lean:, {partisan_lean}')
    print(f'session_id:, {session_id}')
    print(f'chamber_id:, {chamber_id}')

    # Let the queue know that the message is processed
    message.delete()