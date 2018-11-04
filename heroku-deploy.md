# Deploying to Heroku

Congratulations !!! 

![A](https://media.giphy.com/media/srg19CG0cKMuI/giphy.gif)

You have made your first Machine Learning app and would now like to host it somewhere to show it to your friends/families/colleagues what you had been working on for the past few days.

If you are like me who'd rather work on the ML problem instead of worrying about Django or Flask, Heroku or pythonanywhere, then you have come to the right place. 

Now I am no expert in Web development nor Heroku, but I am going to provide you with step by step instructions to take your ML app and deploy it on Heroku for FREE.

Before we begin, a big shoutout to [Simon Willison](https://github.com/simonw), whose code for [cougar-or-not](https://github.com/simonw/cougar-or-not) has been used as the base over which many other enhancements have been made. Currently this uses Starlette ASGI framework and I am thinking of changing it to Flask (since it is more popular) in the future.

So let's get started.

### Step 1: Install Docker CE

A. Your application deployed on Heroku would be running inside a Docker container, so you need to install Docker CE (community edition) locally on your computer.
Follow the instructions given [here](https://docs.docker.com/install/) to install Docker for your version of OS.

B. After installing Docker, add your current user to the docker group using below command. This will ensure that you don't have to type sudo for running any docker or heroku commands. You would have to logout and login again for the new permissions to take effect.
```
sudo usermod -aG docker $USER
```

C. Now check that docker installation is correctly done by running `docker ps` command. If you see any permission issue, then it most likely means you did not re-login.

### Step 2: Download the code
A. Clone the repo: `git clone https://github.com/nikhilno1/healthy-or-not.git`  
B. Change the app name: `mv healthy-or-not <app-name>`  
C. `cd <app-name>`  
D. `mv food-detector.py <app-name>.py`  
E. Make following changes in your `<app-name>.py`
* Change the class name in `classes = ['healthy', 'junk']`
* Specify the correct resnet model in `create-cnn()`. Currently it uses `resnet50`.
* Change the heading of your welcome page  

F. Overwrite the weights file: `cp </path-to-model-weights/file.pth> model-weights.pth`  
G. Open Dockerfile: `vi Dockerfile`  
H. Replace app name: `:%s/food-detector/<app-name>/g`

### Step 3: Verify locally  
Before pushing it to Heroku, it is better to test the app locally. Your application will run inside a docker container.  
A. Build the docker image: `docker image build -t <app-name>:latest`  
B. Run the docker image: `docker run -d --name <app-name> --publish 80:8008 <app-name>:latest`  
C. If you make changes to any file, then you need to rebuild and restart the docker container. This can be done by first running below two commands and then running the above commands:  
   `docker container stop <app-name>`  
   `docker container rm <app-name>`  
   
D. Once the docker container runs, then you can open `http://YOUR_IP/`. If for any reason you cannot use port 80, then use any other port but remember to specify the same in `docker run` command.     
E. Once your app works fine locally, then you can move to the next and last step which is to run it on heroku.

### Step 4: Deploy to Heroku.
A. Create your account on [Heroku](https://www.heroku.com/)  
B. Install Heroku CLI by following the instructions [here](https://devcenter.heroku.com/articles/heroku-cli#download-and-install).  
In brief, 
* Install `git` if not already present.  
* Install snap or snapd (depending on platform). For my debian linux, I installed using following commands:  
   `sudo apt update`  
   `sudo apt install snapd` 
* `sudo snap install --classic heroku`  
I remember my heroku CLI getting installed in non-standard path somwehere in /snapd/, so create a symbolic link if you don't want the hassle of typing the path everytime.  

C. Login to heroku using one of the two methods:  
* `heroku container:login`
* `docker login --username=<your username> --password=$(heroku auth:token) registry.heroku.com`  

D. Create heroku app: `heroku create <app-name>`  
E. Push your local app to heroku: `heroku container:push web -a <app-name>`  
F. Release your app: `heroku container:release web -a <app-name>`  
G. Access your app: `heroku open`. If this doesn't open the browser window, then it will print the URL which you can open from any browser.

That's it. Let me know how it goes. I'll be adding more and more things to the UI so keep checking the repo for any updates. 


