# Incoherent Bullet Synthesizer (IBS)

The goal of this project is to produce "real"-sounding Air Force performance bullet statements using a machine learning model. 

This is an interesting problem for a computer to solve, due to the fact that bullet "best practices" are almost completely arbitrary,
 largely depending on the whims of supervisors, superintendents, first shirts, and commanders. This project isn't intended to be a serious
 tool for people to write their own bullets.. at best, it can maybe help get some creative juices flowing during EPR/OPR/Award season(s)
 when people realize they don't have enough bullets to fill the form.

## Training Dataset

The machine learning model was trained using bullets (mostly EPR) that were found online. A simple web scraper was built to collect all these bullets, which were then
 stored in a flat text file. A link to the web scraper will be placed here once I upload the web scraper to github. The scraper
 is written in Python, with an additional Perl script to organize the data into one big text file. Over 40,000 bullets were collected using the web scraper. 

## Training Method

The machine learning models were built using Tensorflow in Python, using jupyter notebooks. Through a process of trial and error, I ended up creating four models. The models varied in the number of LSTM layers, LSTM neurons, and training epoch sizes, but all used the same training dataset. GTX 2070 and GTX 970 graphics cards, acquired previously via Officer Pay (TM), was used to train the models. The models took around two and a half hours to train each. 

## Front-end

The front-end is based on some previous work I did [here](https://github.com/EA-Pods-Team/bullets-web). In order to get predictions working on the web, I utilized Tensorflow.js and a python script to convert my Tensorflow model into one that Tensorflow.js could read.  

### Explanation of inputs on front-end

#### Lines to generate

Self-explanatory

#### Spiciness

This controls how unpredictable the bullets are. This is usually called "temperature" in machine learning lingo, but I thought spiciness sounded funnier.
 A lower value will generate bullets that are very similar to the 40,000+ training examples, whereas a larger value will be more creative, but may produce gibberish sometimes.

#### Model Selection

I created four models, you will notice that each has its own "quirks" if you run the models enough times...

#### Output Type

In Free Run mode, the primer text is used as the beginning of the first bullet, and then the model will continue creating subsequent bullets with any beginning verb or whatever. This method tends to produce more creative and varied results. The primer text is supposed to "inspire" the output results, but from my limited testing I don't think the primer text does much besides control the beginning of the first bullet.

In Same Beginning mode, each bullet ouputted will have the same prefix. I was imagining that with this mode might be helpful for people that have 2/3rds of a bullet complete and are at a loss for what to end it with.. If they threw it in here, they could get a bunch of creative examples on what they could do.

#### Primer Text

Explained in previous paragraph



