import praw
import requests
from build import *

def determine_breed(url):
	model = load_model()
	if '.jpg' in url:
		print(url)
		image = requests.get(url).content
		with open(submission_image + '/check.jpg', 'wb') as handler:
			handler.write(image)
		submission_data = create_submission_data(url)
		for data in submission_data[:1]:
			image_data = data[0]
			data = image_data.reshape(size,size,1)
			model_out = model.predict([data])[0]
			if np.argmax(model_out) == 0: 
				dog_breed = 'Labrador Retriever'
			elif np.argmax(model_out) == 1:
				dog_breed = 'German Shepherd'
			elif np.argmax(model_out) == 2:
				dog_breed = 'Bulldog'
			elif np.argmax(model_out) == 3:
				dog_breed = 'Beagle'
			elif np.argmax(model_out) == 4:
				dog_breed = 'Poodle'
			elif np.argmax(model_out) == 5:
				dog_breed = 'Rottweiler'
			elif np.argmax(model_out) == 6:
				dog_breed = 'Yorkshire Terrier'
			elif np.argmax(model_out) == 7:
				dog_breed = 'German Shorthaired Pointer'
			elif np.argmax(model_out) == 8:
				dog_breed = 'Siberian Husky'
			elif np.argmax(model_out) == 9:
				dog_breed = 'Pembroke Welsh Corgi'
			elif np.argmax(model_out) == 10:
				dog_breed = 'Dachshund'
			elif np.argmax(model_out) == 11:
				dog_breed = 'Australian Shepherd'
			elif np.argmax(model_out) == 12:
				dog_breed = 'Miniature Schnauzer'
			elif np.argmax(model_out) == 13:
				dog_breed = 'Boxer'
		os.remove(submission_image + '/check.jpg')
		return dog_breed

def authenticate():
	reddit = praw.Reddit(client_id='CLIENT_ID',
                        client_secret='CLIENT_SECRET',
                        password='PASSWORD',
                        user_agent='Doggit by /u/Lvl1Stantler',
                        username='DoggitBot')
	return reddit


#Start Bot
reddit = authenticate()
print(reddit.user.me())
while True:
	print('Loop')
	subreddit = reddit.subreddit('rarepuppers').new()
	for submission in subreddit:
		submission.comments.replace_more(limit=None)
		comment_queue = submission.comments[:]
		while comment_queue:
			comment = comment_queue.pop(0)
			if '!breed' in comment.body:
				print('Found')
				comment.reply('The dog in the picture appears to be a ' + determine_breed(submission.url) + '.')
			comment_queue.extend(comment.replies)

	subreddit = reddit.subreddit('allthingsdogs').new()
	for submission in subreddit:
		submission.comments.replace_more(limit=None)
		comment_queue = submission.comments[:]
		while comment_queue:
			comment = comment_queue.pop(0)
			if '!breed' in comment.body:
				print('Found')
				comment.reply('The dog in the picture appears to be a ' + determine_breed(submission.url) + '.')
			comment_queue.extend(comment.replies)

	subreddit = reddit.subreddit('dogpictures').new()
	for submission in subreddit:
		submission.comments.replace_more(limit=None)
		comment_queue = submission.comments[:]
		while comment_queue:
			comment = comment_queue.pop(0)
			if '!breed' in comment.body:
				print('Found')
				comment.reply('The dog in the picture appears to be a ' + determine_breed(submission.url) + '.')
			comment_queue.extend(comment.replies)


