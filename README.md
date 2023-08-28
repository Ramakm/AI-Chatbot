# **AI-CHATBOT**
This repo is about an AI Chatbot using Python and Flask REST API<br>

![screen-capture](https://github.com/Ramakm/AI-Chatbot/assets/8182816/83b7fd0a-21c7-4889-b4bf-43dd5420da91)


# **STEPS TO FOLLOW:**

Please follow each and every steps mentioned in this README.md file properly. You will definitely able to run the whole project.
If you find out any error, please raise an issue here, I will try to solve your error as soon as possible.

## 1. Requirements (libraries)

* **TensorFlow:** TensorFlow is an open-source machine learning framework developed by Google. It provides tools and libraries for building and training various machine learning models, including neural networks.

* **Flask:** Flask is a micro web framework for Python. It is used to build web applications, including RESTful APIs, in a simple and lightweight manner.

## 2. VS Code Setup:

I did this project in VS Code. If you have anything else like Pycharm and Conda. That also fine but be sure about `PATH` and all.

* **Clone the Repository:** This refers to making a copy of the code repository (project) from a remote repository (usually hosted on a platform like GitHub) onto your local machine.
  You can directly download the zip file from this repo and extract the files and drag & drop to your new window in VS code.

* **Create a Python Virtual Environment:** A virtual environment is an isolated Python environment in which you can install packages without affecting your system-wide Python installation.      This helps to keep dependencies for different projects separate.
```
# macOS/Linux
# You may need to run sudo apt-get install python3-venv first
python3 -m venv .venv

# Windows
# You can also use py -3 -m venv .venv
python -m venv .venv
```
* **Activate the Virtual Environment:** Activating the virtual environment allows you to work within the isolated environment, ensuring that the packages you install are specific to this project and won't interfere with other projects or your system Python.
```
#linux
source ./venv/bin/activate  # sh, bash, or zsh

#windows
.\venv\Scripts\activate
```
* **Install TensorFlow:** This step is about installing TensorFlow, the machine learning framework that the chatbot will use for training and prediction.
  ```
  pip install tesorflow
  ```

* **Install NLTK:** NLTK (Natural Language Toolkit) is a Python library for working with human language data. It's used here for text tokenization and other NLP tasks.
  ```
  pip install nltk
  ```
* **Install Flask:** Flask is needed to create a web server and API for the chatbot. It will handle incoming user messages and provide responses.
  ```
  pip install flask
  ```
* **Install Flask-Ngrok (Optional):** Ngrok is a tool that allows you to expose your local web server to the internet. This step is optional and is used for easily sharing your chatbot with others online.
  ```
  pip install flask-ngrok
  ```

* **Configure Ngrok Credentials:** If you decide to use Ngrok, you'll need to set up an account and provide your credentials for authentication. So, instead of doing that, I would suggest ignore the optional work. To expose your bot via Ngrok, run ```pip install flask-ngrok``` to install ```flask-ngrok``` Then you'll need to configure your ngrok credentials(login: email + password) Then uncomment this line ```run_with_ngrok(app) ``` and comment the last two lines ```if __name__ == "__main__": app.run() ```

## **3. Execution:**

* Firstly, Just delete the existing `chatbot_model.h5` file from the folder.
* Then, run the ```train.py``` file to train the model. This will generate a file named ```chatbot_model.h5```. You will face error becauz if you haven't changed the path of the 'intents.json'
  file path in `datafile` variable.
* This is the model which will be used by the Flask REST API to easily give feedback without the need to retrain.
* After running ```train.py```, next run the ```app.py``` to initialize and start the bot.
* To add more terms and vocabulary to the bot, modify the ```intents.json``` file and add your personalized words and retrain the model again.
* Accessing the Chatbot: After running the `Flask app (app.py)`, you can access the chatbot by navigating to ```http://127.0.0.1:5000/``` in your web browser. If you're using Ngrok, you'll access the chatbot through the Ngrok-generated URL.
![image](https://github.com/Ramakm/AI-Chatbot/assets/8182816/679576fe-14b0-4a0c-af8f-ff234fb10922)


# Blog Related To This Project:

I have wrote a whole detailed blog related to this project. Check it out:

[Build An AI Chatbot From Scratch](https://medium.com/@ramakrushna_mohapatra8594/create-an-ai-chatbot-from-scratch-738ea385d108)

<!-- Actual text -->
# **Find me on**
[![LinkedIn](https://img.icons8.com/color/48/000000/linkedin.png)](https://www.linkedin.com/in/ramakrushnamohapatra/)
[![Twitter](https://img.icons8.com/color/48/000000/twitter.png)](https://twitter.com/codewith_ram)

## **Having troubles implementing?**

Hit an issue bottom in this repo or else reach out to me via LinkedIN or Twitter message. I will try to reply to you as soon as possible.

Hopefully You will able to run this one. **Give it a `star` and `fork` this repo**.


<a href="https://www.buymeacoffee.com/Ramakrushna" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

@copyright reserved Ramakrushna 2023
