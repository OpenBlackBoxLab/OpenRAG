from azure.storage.queue import QueueClient

class AzureQueueHandler:
    def __init__(self, connection_string, queue_name):
        self.queue_client = QueueClient.from_connection_string(conn_str=connection_string, queue_name=queue_name)
        try:
            self.queue_client.get_queue_properties()
        except Exception as e:
            self.queue_client.create_queue()

    def send_message(self, message):
        sent_message = self.queue_client.send_message(message)
        return sent_message

    def peek_messages(self, max_messages=5):
        peeked_messages = self.queue_client.peek_messages(max_messages=max_messages)
        return peeked_messages

    def receive_messages(self, max_messages=5):
        messages = self.queue_client.receive_messages(messages_per_page=max_messages)
        return messages

    def delete_message(self, message):
        self.queue_client.delete_message(message)

    def update_message(self, message, content, visibility_timeout=0):
        self.queue_client.update_message(message, content=content, visibility_timeout=visibility_timeout)

    def clear_queue(self):
        self.queue_client.clear_messages()