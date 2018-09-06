# Doggit
A Reddit bot that analyzes image submissions and determines which dog breed is in the picture. The bot's username is /u/DoggitBot and is monitoring the subreddits /r/rarepuppers, /r/allthingsdogs, and /r/dogpictures. Doggit currently only supports 14 different dog breeds. 

## Using Doggit
Using Doggit is extremely fast and simple, since all you need is a functioning Reddit account!
To use Doggit, go on one of the monitored subreddits and comment the following on an image submission:
```
!breed
```
/u/DoggitBot should then respond to your post with one of the 14 supported breeds.

## Built With
* [Reddit API](https://praw.readthedocs.io/en/latest/index.html) - Interface that was used
* [TFLearn](http://tflearn.org/) - Used to create the convolutional neural network

## Authors
* **Stanley Tran** - [Stantl3r](https://github.com/Stantl3r/Doggit)
