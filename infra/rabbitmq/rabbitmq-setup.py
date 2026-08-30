import pika

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.exchange_declare(exchange='pipeline', exchange_type='topic', durable=True)

channel.queue_declare(queue='ai_tailor_queue', durable=True)
channel.queue_bind(exchange='pipeline', queue='ai_tailor_queue', routing_key='ai_tailor_queue')

channel.queue_declare(queue='smpl_anthro_queue', durable=True)
channel.queue_bind(exchange='pipeline', queue='smpl_anthro_queue', routing_key='mesh.ready')

channel.queue_declare(queue='anny_queue', durable=True)
channel.queue_bind(exchange='pipeline', queue='anny_queue', routing_key='measurements.ready')

print("[INFO] Exchange and queues set up.")
connection.close()
