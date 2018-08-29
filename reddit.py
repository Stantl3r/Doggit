import praw
import requests
from build import *

def determine_breed(url):
	model = load_model()
	if '.jpg' in url:
		print('Found')
		image = requests.get(url).content
		with open(submission_image + '/check.jpg', 'wb') as handler:
			handler.write(image)
		submission_data = create_submission_data(url)
		for data in submission_data[0]:
			image = data[0]
			image_data = image.reshape(size,size,1)
			model_out = model.predict([data])[0]
			if np.argmax(model_out) == 0: 
				dog_breed = 'Labrador'
			elif np.argmax(model_out) == 1:
				dog_breed = 'German'
			elif np.argmax(model_out) == 2:
				dog_breed = 'Bulldog'
			elif np.argmax(model_out) == 3:
				dog_breed = 'Beagle'
			elif np.argmax(model_out) == 4:
				dog_breed = 'Poodle'
			elif np.argmax(model_out) == 5:
				dog_breed = 'Rottweiler'
			elif np.argmax(model_out) == 6:
				dog_breed = 'Yorkshire'
			elif np.argmax(model_out) == 7:
				dog_breed = 'Pointer'
			elif np.argmax(model_out) == 8:
				dog_breed = 'Husky'
			elif np.argmax(model_out) == 9:
				dog_breed = 'Corgi'
			elif np.argmax(model_out) == 10:
				dog_breed = 'Dachshund'
			elif np.argmax(model_out) == 11:
				dog_breed = 'Australian'
			elif np.argmax(model_out) == 12:
				dog_breed = 'Schnauzer'
			elif np.argmax(model_out) == 13:
				dog_breed = 'Boxer'
		print(dog_breed)
		for image in os.listdir(submission_image):
			os.remove(image)


reddit = praw.Reddit(client_id='***REMOVED***',
                     client_secret='***REMOVED***',
                     password='***REMOVED***',
                     user_agent='testscript by /u/DoggitBot',
                     username='DoggitBot')

subreddit = reddit.subreddit('test').new()
for submission in subreddit:
	submission.comments.replace_more(limit=None)
	comment_queue = submission.comments[:]
	while comment_queue:
		comment = comment_queue.pop(0)
		print(comment.body)
		"""if '!breed' in comment.body:
			determine_breed(submission.url)"""
		determine_breed(submission.url)
		comment_queue.extend(comment.replies)


