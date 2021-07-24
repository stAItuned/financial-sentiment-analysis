# Financial Sentiment Analysis


<p align="center">
  <img height="270" src="demo/img1.jpg">
  <img height="270" src="demo/img2.jpg">
</p>

| **License** | **Language** | **Libraries** |
| ----- | ---- | ---- |
| ![Licence](https://img.shields.io/badge/Licence-MIT-orange) |  ![Python](https://img.shields.io/badge/Python-yellow)| ![Pytorch](https://img.shields.io/badge/Pytorch-1.8.1-brightgreen) ![Flask](https://img.shields.io/badge/Flask-2.0.0-brightgreen) ![Streamlit](https://img.shields.io/badge/Streamlit-0.82.0-brightgreen) ![Pandas](https://img.shields.io/badge/Pandas-1.2.4-brightgreen) 


### Contents
- [Description](#description)
- [Dataset](#dataset)
- [Structure](#structure)
- [Solutions](#solutions)
- [How to run it](#howtorun)
- [How to contribute](#howtocontribute)
- [Demo](#demo)
- [Authors](#authors)

------------------------

<a name="description"/>

## Description
The goal of this project is to give you the chance of looking at the **dependence** between the **news sentiment** of a company with respect to its **share price**. 
*This project is one part of a main project still on its way, which aims to provide a financial analysis platform.*

<a name="dataset"/>

## Dataset
News data are taken from the [polyglon AIP](https://polygon.io/), using the *stocks basic API plan* account.

<a name="structure"/>

## Structure
The project structure consists on a **server side**, which manages the calls to the sentiment analysis model. The server provides a **service API** for the inference of the model. 

<a name="solutions"/>

## Solutions
We developed the *sentiment analysis model* as a **neural network model**, based on **fine-tuning** of an existent pre-trained model. We trained and experimented different models, until we reached the **best one**. In order to keep our model **private**, we provide **just** a pre-trained model as a sentiment analysis model, that is the **transformers** model taken from *hugging face*

------------------------
<a name="howtorun"/>

## How to run 
0. Create a new virtual environment
```
python -m venv ./venv
source venv/bin/activate
pip install --upgrade pip
```
1. Clone the repository and move to the downloaded directory
```
git clone https://github.com/sicilian-scientists/financial-sentiment-analysis
cd financial-sentiment-analysis
```
2. Request [Polyglon's API key]() (FREE)
3. Write the given key in 'files/api_key_polyglon.txt' 
```
echo API_KEY > files/api_key_polyglon.txt
```
4. Install the requirements
```
pip install requirements.txt
```
5. Run local server and dashboard
```
python api_server.py
streamlit run main.py
```
6. You're ready to go! 

<a name="howtocontribute"/>

## How to contribute
1. Fork the repository
2. Do the desired changes
3. Make a pull request

<a name="demo" />

## Demo (GIF)
<p align="center">
  <img src="demo/demo.gif" height="50%"/>
</p>


<a name="authors"/>

#### Authors

[![Linkedin](https://img.shields.io/badge/Linkedin-Daniele%20Moltisanti-blue)](https://www.linkedin.com/in/daniele-moltisanti/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Daniele%20Moltisanti-9cf)](https://daniele21.github.io)


[![Linkedin](https://img.shields.io/badge/Linkedin-Francesco_Di_Salvo-blue)](https://www.linkedin.com/in/francescodisalvo-pa/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Francesco_Di_Salvo-9cf)](https://francescodisalvo05.github.io/)
