from queue import Queue

class Synchronize(object):
    def __init__(self, topics, sync_method, trigger_topic, verbose=True):
        self.topics = topics
        self.sync_method = sync_method
        self.trigger_topic = trigger_topic
        self.verbose = verbose
        
        self.sync_topics = self._create_topic_queues(topics)

    def _create_topic_queues(self, topics):
        sync_topics = []
        for topic, info in topics.items():
            if not self.topics[topic]['sync']:
                continue
            sync_topics.append(topic)
            topic_queue = Queue()
            setattr(self, topic, topic_queue)
        return sync_topics

    def synchronize(self, topic, msg):
        #1 Add message to queue
        topic_queue = getattr(self, topic)
        topic_queue.put(msg)

        if self.verbose:
            self._print_queue_sizes()
            # print("Sync message header ", msg.header)
            print(f'Topic {topic} timestamp: {msg.header.stamp.to_sec()}')
        
        #2 Correct frames with dropped packets
        if self.sync_method == 'HARDWARE_SYNC_METHOD':
            self._hardware_sync()
        else: 
            raise NotImplementedError

        #3 Synchronize frames if all topics have a message and return sync dict
        if self._all_topics_have_message():
            sync_dict = self._create_sync_dict()
            return sync_dict
        else:
            return None

    def _create_sync_dict(self):
        """
        Creates a dictionary with synchronized messages
        """
        sync_dict = {
            'ts': None,
            'topics': {}
        }
        for topic in self.sync_topics:
            topic_queue = getattr(self, topic)
            msg = topic_queue.get()
            if topic==self.trigger_topic:
                sync_dict['ts'] = msg.header.stamp
            sync_dict['topics'][topic] = msg
        
        return sync_dict
    
    def _print_queue_sizes(self):
        """
        Prints the size of each queue
        """
        for topic in self.sync_topics:
            topic_queue = getattr(self, topic)
            print(f'{topic} queue size: {topic_queue.qsize()}')

    def _all_topics_have_message(self):
        """
        Checks if all topics have a message in their queue
        """
        for topic in self.sync_topics:
            topic_queue = getattr(self, topic)
            if topic_queue.empty():
                return False
        return True

    def _hardware_sync(self):
        """
        This method assumes assumes that the trigger sensor is first each set S. 

        1: If trigger topic has no frames, trigger topic must have been dropped. Therefore drop all received topics until trigger topic is received
        2: If trigger topic has more than 1 frame, non-trigger topics must have dropped at least one frame
            Remove all topics with timestamps older than latest trigger topic
        3: Trigger topic is published assumption is violated!
        """
        # Case 1:
        if getattr(self, self.trigger_topic).empty():
            for topic in self.sync_topics:
                topic_queue = getattr(self, topic)
                while topic_queue.qsize() >= 2:
                    print(f'Dropped {topic} frame')
                    topic_queue.get()
    
        # Case 2:
        elif getattr(self, self.trigger_topic).qsize() > 1:
            while getattr(self, self.trigger_topic).qsize() > 1:
                getattr(self, self.trigger_topic).get() # Pop erroneous sets

            last_trigger_time = getattr(self, self.trigger_topic)[0].header.stamp.to_sec()

            for topic in self.sync_topics:
                topic_queue = getattr(self, topic)
                msg = topic_queue[0]

                if msg.header.stamp.to_sec() < last_trigger_time:
                    print(f'Dropped {topic} frame')
                    topic_queue.get()
                    import pdb; pdb.set_trace()