import boto3


sqs = boto3.resource("sqs", region_name='us-east-1')
queue = sqs.get_queue_by_name(QueueName="mlgovhawk-test")

df_to_process = None

def process_message(message):
    if message.message_attributes is not None:
        print(message.attributes.get('revision_id').get('StringValue'))
    else:
        print("No Attributes.")


messages = queue.receive_messages()
for message in messages:
    print(message)
    try:
        process_message(message)
    except Exception as e:
        print(f"exception while processing message: {repr(e)}")
        continue

    message.delete()