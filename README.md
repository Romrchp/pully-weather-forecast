# Will it rain tomorrow in Pully ?

This Machine Learning project's objective is to predict the weather in Pully (Switzerland), based on weather-related parameters in different cities , measured during an entire day. This specific project was done entirely using Julia.

For an exhaustive description of the project's motivation, implementations & results, please feel free to read our 📖 [rainy_report.pdf](rainy_report.pdf).


## Repository Structure :

- 📒 [rainy_book.jl](rainy_book.jl) : The Pluto notebook in which our entire code is located, from the transformation of the input data to the tuning of our different models.

- 📖 [rainy_report.pdf](rainy_report.pdf) : Report for our project which contains a complete description of our project. Our choices, the methods and the reasons for their use as well as our interpretation of the different results we obtained are all gathered in it.

- 💾 [trainingdata.csv](trainingdata.csv) : The training data file we were given, in a CSV format. This training set was transformed throughout the project in order to train correctly our models.

- 💾 [testdata.csv](testdata.csv) : The data set (in CSV format) for which we needed to predict whether the values contained, in each row, corresponds to rain in pully the day after, or not. Modified as well to match the transformations made on the training set.

## Reproducibility :

In another to read another dataset than the two CSV files detailed above, the file containing this data must be under a CSV format and located in the same directory as the 📒 [rainy_book.jl](rainy_book.jl) notebook. The name of the file must then be replaced in the `CSV.read(joinpath(FileName.csv), DataFrame)` of the notebook.

Running the notebook cells will then produce results files. These files were to be uploaded to the Kaggle page of our course, to compare with the hidden expected results.

## Team Members :

- [Alysée Khan](mailto:alysee.khan@epfl.ch) 
- [Romain Rochepeau](mailto:romain.rochepeau@epfl.ch) 