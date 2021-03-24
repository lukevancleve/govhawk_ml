import boto3

import os
import pandas as pd
from src.models.predict_model import make_predictions

if __name__ == '__main__':
    # Get the service resource
    sqs = boto3.resource('sqs', region_name='us-east-1')

    # Get the queue
    queue = sqs.get_queue_by_name(QueueName='mlgovhawk-test')


    df = None

    # Process messages by printing out body and optional author name
    found_last_message = False
    while not found_last_message:

        messages = queue.receive_messages(MessageAttributeNames=['All'],
                                            MaxNumberOfMessages=10)

        if len(messages)==0:
            print("Read all messages:")
            break

        for message in messages:
            
            # Get the custom author message attribute if it was set
            revision_id = ''
            if message.message_attributes is not None:
                bill_version_id = message.message_attributes.get('bill_version_id').get('StringValue')
                bill_id = message.message_attributes.get('bill_id').get('StringValue')
                version_number = message.message_attributes.get('version_number').get('StringValue')
                partisan_lean = message.message_attributes.get('partisan_lean').get('StringValue')
                session_id = message.message_attributes.get('session_id').get('StringValue')
                chamber_id = message.message_attributes.get('chamber_id').get('StringValue')
                plain_url = message.message_attributes.get('plain_url').get('StringValue')
            
            # Print out the body and author (if set)
            print(f'bill_version_id:, {bill_version_id}')
            print(f'bill_id:, {bill_id}')
            print(f'version_number:, {version_number}')
            print(f'partisan_lean:, {partisan_lean}')
            print(f'session_id:, {session_id}')
            print(f'chamber_id:, {chamber_id}')
            print(f'plain_url:, {plain_url}')

            row = pd.DataFrame.from_records([
                            {'bill_version_id': bill_version_id,
                                'bill_id': bill_id,
                                'version_number': version_number,
                                'partisan_lean': partisan_lean,
                                'session_id': session_id,
                                'chamber_id': chamber_id,
                                'plain_url': plain_url    
            }])
            
            # If any information is missing, write it to an error log
            if row.isnull().values.any():
                error_log = open('error_log.text', 'a')
                error_log.write(row.__str__)
                continue
                

            if df is None:
                df = row
            else:
                df = pd.concat([df,row], axis=0)


            # Let the queue know that the message is processed
            message.delete()


    if df is not None:

        preds = make_predictions_from_sqs(df)
        print(preds)


        # Write this to a table somewhere.
