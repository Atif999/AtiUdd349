# ML-DEPLOY

**Prerequisites**
Please have docker installed on your system

**Steps to run the project**

1. Clone the repo(git clone https://github.com/Atif999/ML-DEPLOY.git) on your system
2. Go to the directory where repo cloned and go to the terminal
3. Run the command `docker-compose up` to build and start the container
4. Once container is built and started you can access the REST API of ML model deployed on localhost:8000/predict
5. You can predict label based on a json input of {"text":"sample text"} here the sample text be the german text you want prediction against
6. Once container up and running you can access the api docunentaion at localhost:8000/docs
