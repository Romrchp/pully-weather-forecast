### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 9ff5b5bc-eb0f-456b-852e-22004fb52634
begin
    using Pkg
	Pkg.activate(joinpath(Pkg.devdir(), "MLCourse"))
	Pkg.add("MLJFlux")
	Pkg.add("MLJ")
	Pkg.add("EarlyStopping")
	Pkg.add("TSne")
    using LinearAlgebra, Random, MLCourse
    import MLCourse: PolynomialRegressor, poly
end


# ╔═╡ cc54ecc2-cfcb-4096-b5c4-d58f1e91b14d
using EarlyStopping, TSne

# ╔═╡ a4d202fa-c71e-4599-ad15-2874f95ea98b
using StatsPlots, Distributions

# ╔═╡ a88c2ef8-e16c-4db0-b826-16d7ab86e0dc
using MLJMultivariateStatsInterface

# ╔═╡ fbfd2af3-b31f-4674-9854-4f60cb10c6ae
using NearestNeighborModels

# ╔═╡ 420cb2b8-ce8e-4b82-abc1-3f16da357982
using OpenML, MLJ, MLJXGBoostInterface, DataFrames, MLJLinearModels, MLJDecisionTreeInterface, CSV, Plots

# ╔═╡ 16be29f3-31ae-4fe9-bb4c-b4cb17d019bc
begin
	using MLJFlux, Flux
	Core.eval(Main, :(using MLJ))
end

# ╔═╡ 30c875e8-ffea-4391-91e7-a6fbbc639cd8
Random.seed!(2);

# ╔═╡ 4d2179df-cbf9-42dd-b6b0-021a98d29259
md"# ML Project

## Raw data inspection and transformation

First, let's look at the raw data. 
"

# ╔═╡ d67c9d59-53ff-4091-a810-13490ca7c3db
CSV_Training_Data=CSV.read(joinpath("trainingdata.csv"), DataFrame);

# ╔═╡ af2619ab-b851-4f0a-9a81-03b513cbebca
CSV_Test_Data=CSV.read(joinpath("testdata.csv"), DataFrame);

# ╔═╡ 5737aa4e-dd68-449a-9909-156a2ed74519
md"
First, we make sure that our data are Float and that the prevision data is multiclass.
"

# ╔═╡ e24d2e2d-381a-4777-af79-49aa44d4e374
begin
	coerce!(CSV_Training_Data, Count => MLJ.Continuous)
	coerce!(CSV_Training_Data, :precipitation_nextday => Multiclass)
	coerce!(CSV_Test_Data, Count => MLJ.Continuous)
end;

# ╔═╡ 058da5bf-a59b-4e00-9698-ff3d0852d421
md"
The first aim is to look if missing values are present in the CSV TrainingData set :
"

# ╔═╡ 59fa7078-7b4f-4984-b676-698d6ffc496d
any(ismissing.(Array(CSV_Training_Data)))

# ╔═╡ 1cd896e7-70e0-4178-992d-7b2e1ece1839
md"
We do have missing values. Dropout those missing values leads to the dropping of nearly half of the original number of rows. The loss of information is therefore way to important if we perform a dropmissing on our DataSet.

We thus choose to replace the missing values with the mean values of the column, as even if values are maybe not exactly what they are supposed to be, it's still better than loosing half of the data!

Same operation is done on the test data as well.
"

# ╔═╡ 136c3257-f70f-4acb-8fbb-316ea4578835
begin
	Without_LC = select(CSV_Training_Data, Not(:precipitation_nextday));
	Training_Data = MLJ.transform(fit!(machine(FillImputer(), Without_LC)), Without_LC)
	Test_Data= MLJ.transform(fit!(machine(FillImputer(), CSV_Test_Data)), CSV_Test_Data)
end;

# ╔═╡ c5984f1e-28d5-4b45-8d86-c0b0142579f8
md"
In order to obtain the 'cleanest' data possible, we create two function that will be necessary in order to clean the data.
- BigChop will replace the very large values making no-sense (larger than 1000 in absolute value) by missing, therefore we will be able to change them by the mean of the column afterwards. As there are very few of them, replacing the value by the mean of the columns should not add a lot of noise. In addition, this could be better for the accuracy of the treatment we also do to the test data.
- chop will put the small values (smaller than 10^-8 in absolute value) to 0. This is done in order to reduce noise.

Code duplication for the use of the MLJ.Transform function is unfortunate, but as chop & BigChop can not work properly on missing values, we don't get many other choices.
"

# ╔═╡ 6ec9e33f-8e37-43b6-ad10-38d0c3f65e58
BigChop(x, eps = 1000) = abs(x) < eps ? x : missing;

# ╔═╡ 3f3a2c4a-a6a9-4d4a-a8f7-2d9fe9aa2492
chop(x, eps = 1e-8) = abs(x) > eps ? x : zero(typeof(x));

# ╔═╡ 298772c2-01e3-4ea7-854b-1afc4fa7a054
md"

In addition to using both these functions, as the culumns :ZERsunshine1 and :ALTsunshine4 are set to a constant (0), they do not give any new information. Therefore, we remove them.
"

# ╔═╡ f75efce8-2a3c-4f50-b475-23b72fefd654
begin
	BC_Training = BigChop.(Training_Data)
	Clean_Data =chop.(select(MLJ.transform(fit!(machine(FillImputer(), BC_Training)), 
    BC_Training), Not([:ZER_sunshine_1, :ALT_sunshine_4])))
end;

# ╔═╡ adb61dc7-ddfd-4c39-8924-264b992461fe
md"
As we chopped the training data and removed it two columns, we also chop the test data and remove the two same columns from it.
"

# ╔═╡ cf6d997f-5aa1-4b5e-bedd-581c27a71d58
begin
	BC_Test= BigChop.(Test_Data)
	Clean_Test_Data = chop.(select!(MLJ.transform(fit!(machine(FillImputer(), BC_Test)), BC_Test), Not([:ZER_sunshine_1, :ALT_sunshine_4])))
end;

# ╔═╡ 526f11c6-045b-4fbe-8cb7-61d7c4e73ebd
Data_precipitation=DataFrame(precipitation_nextday
					= CSV_Training_Data.precipitation_nextday)[:,1]

# ╔═╡ 9955afa5-413d-4de5-9d25-d3f04710764c
md" Now, we decide to standardize the data.
"

# ╔═╡ d30f7886-f509-48ca-93d5-447a56e75b73
Standardized_Data = DataFrame(MLJ.transform(fit!(machine(Standardizer(), Clean_Data)), Clean_Data));

# ╔═╡ d86955c2-e84b-46f6-9df1-93fd5602ccef
md"In the case of the test set, three additional columns have a constant value set to 0. Therefore, we do not standardize those ones.
"

# ╔═╡ c376351c-8373-46f9-8180-e1510060f0f8
begin
	Standardized_test_mach = fit!(machine(Standardizer(features=[:ABO_sunshine_4, :CHU_sunshine_4, :SAM_sunshine_4], ignore=true), Clean_Test_Data))
	Final_Test_Data = DataFrame(MLJ.transform(Standardized_test_mach, Clean_Test_Data))
end;

# ╔═╡ 521c187d-f983-4d31-8b1c-c0e2f88b9471
md"
Now that the data are a bit more clean and at a comparable scale, we can check if the data are correlated or by which component the variance is the most explained. To do so, we run PCA on our training data.
"

# ╔═╡ 67e5045c-6711-463a-8670-0d70e318ffc7
let
	Mach_Correlation=fit!(machine(@pipeline(Standardizer(), PCA()), Clean_Data))
	gr()
	p1 = biplot(Mach_Correlation)
    p2 = biplot(Mach_Correlation, pc = (1, 3))

    plot(p1, p2, layout = (1, 2), size = (2500, 1000))
end


# ╔═╡ f706d246-7f28-4018-9240-d00a2cfeac7f
md"We see that there is not a global correlation between the different features, we cannot set a general tendancy. The variance is well explained at least until the third component, which was indeed expected. It seems like some features are clearly more explained by the first, second or third component, but again there is no general tendancy making the variance to be well explained only by one component. To find out a bit more about the correlation, we select some features and run correletion plots."

# ╔═╡ 004aa8fb-ea81-4458-80af-f6c7d182f4fc
@df Standardized_Data corrplot([:BAS_radiation_1 :BAS_delta_pressure_1 :BAS_air_temp_1 :BAS_sunshine_1 :BAS_wind_1 :BAS_wind_direction_1],
                     grid = false, fillcolor = cgrad(), size = (1200, 1100))

# ╔═╡ 30b08bdc-09da-4780-aae8-d1624559a6a4
md" We use the correlation plot above to see if the features are corrleated at a given time in a given city.
	We see that some of them, like the radiation with the sunshine, seem to be correlated. But the wind speed and direction seem not to be as much correlated, nor the air temperature with the wind direction for example."

# ╔═╡ 4cc568f3-d1a1-45cd-a793-e4e518d967d8
@df Standardized_Data corrplot([:CDF_wind_1 :CDF_wind_2 :CDF_wind_3 :CDF_wind_4],
                     grid = false, fillcolor = cgrad(), size = (1000, 800))

# ╔═╡ 25579abf-ab22-44dd-b052-cf5b07d1a72c
md" Using PCA again, we indeed see that the wind speed in a given city is correlated over time. The same conclusion can be made for the radiation and the sunshine in a given city at a given time. Thanks to the second plot (rightest), the radiation and sunshine seem to be corraleted at time 3 between BAS and PIO and at time 1 between ENG and ALT. Therefore, we can think that there are some correlations between some cities and features."

# ╔═╡ 114e4c22-b71f-4208-b543-a2b259941a38
let
	Wind_Speed=DataFrame(ENG_wind_1=Clean_Data.ENG_wind_1,ENG_wind_2=Clean_Data.ENG_wind_2, ENG_wind_3=Clean_Data.ENG_wind_3, ENG_wind_4=Clean_Data.ENG_wind_4,
						SHA_wind_1=Clean_Data.SHA_wind_1,SHA_wind_2=Clean_Data.SHA_wind_2, SHA_wind_3=Clean_Data.SHA_wind_3, SHA_wind_4=Clean_Data.SHA_wind_4,
						PIO_wind_1=Clean_Data.PIO_wind_1,PIO_wind_2=Clean_Data.PIO_wind_2, PIO_wind_3=Clean_Data.PIO_wind_3, PIO_wind_4=Clean_Data.PIO_wind_4,
		
	PUY_wind_1=Clean_Data.PUY_wind_1,PUY_wind_2=Clean_Data.PUY_wind_2, PUY_wind_3=Clean_Data.PUY_wind_3, PUY_wind_4=Clean_Data.PUY_wind_4)

Pre_rad=DataFrame(ENG_radiation_1=Clean_Data.ENG_radiation_1,ENG_sunshine_1=Clean_Data.ENG_sunshine_1, PIO_radiation_3=Clean_Data.PIO_radiation_3, PIO_sunshine_3=Clean_Data.PIO_sunshine_3,
						ALT_radiation_1=Clean_Data.ALT_radiation_1,ALT_sunshine_1=Clean_Data.ALT_sunshine_1, BAS_radiation_3=Clean_Data.BAS_radiation_3, BAS_sunshine_3=Clean_Data.BAS_sunshine_3)
	
	Mach_Correlation1=fit!(machine(@pipeline(Standardizer(), PCA()), Wind_Speed))
	Mach_Correlation2=fit!(machine(@pipeline(Standardizer(), PCA()), Pre_rad))
	gr()
	p1 = biplot(Mach_Correlation1)
	p2 = biplot(Mach_Correlation2)
    plot(p1, p2,layout = (1, 2), size = (1000, 600))

end


# ╔═╡ 74d061c3-c0e4-43fa-a9dd-07f59c24a2b3
md"Over time, we see that some features like the wind speed seem to be highly correlated. This is not the case for the wind direction, as plotted below."

# ╔═╡ fa5b2000-4541-4f12-89ae-6471bed07883
@df Standardized_Data corrplot([:CDF_wind_direction_1 :CDF_wind_direction_2 :CDF_wind_direction_3 :CDF_wind_direction_4],
                     grid = false, fillcolor = cgrad(), size = (1000, 800))

# ╔═╡ 3e0c41a1-91ac-4d3a-a907-9790507e63e1
md" Therefore and as mentioned above with the PCA plot, we cannot set a general rule of correlation between the features or for the same feature over time. We can go a step further and also check how the standardized data we will use look like.
"

# ╔═╡ 26843343-cf84-4a58-a158-731bdb27ba40
begin
	Data = MLJ.transform(fit!(machine(PCA(maxoutdim = 2), Standardized_Data)), Standardized_Data)
	scatter(Data.x1, Data.x2,
					 legend = false, c = Int.(int(Data_precipitation)),
					 title = "Standardized data", xlabel = "x1", ylabel = "x2")
end

# ╔═╡ 61b96210-d752-4980-b208-325df173a7f1
md" We see that the variance goes on the x1 direction. But we can not see defined clusters in the data set. Therefore, we use TSne.
"

# ╔═╡ 14d9e84f-f146-45ff-9cd0-81048b415db6
begin
	Random.seed!(2);
	tsne_proj = tsne(Array(Standardized_Data), 2, 50, 1000, 20)
	scatter(tsne_proj[:, 1], tsne_proj[:, 2],
	        c = Int.(int(Data_precipitation)),
			title = "Standardized data",
	        xlabel = "TSne 1", ylabel = "TSne 2",
	        legend = false)
end

# ╔═╡ 3777aa73-65a0-42a4-a9c4-833d71e637cb
md"We see that they might be two clusters, one of true and one of false. Eventhough, the decision boudary seems unclear."

# ╔═╡ 247918e3-afc6-4a12-b5ff-24d52914aa44
md" As the data are maybe a bit noisy, we can use PCA to denoise them.
"

# ╔═╡ 894db97e-07c4-4da0-a443-a4b6d3effb85
let
	pca_data = fit!(machine(PCA(), Standardized_Data));
    vars = report(pca_data).principalvars ./ report(pca_data).tvar
    p1 = plot(vars, label = nothing, yscale = :log10,
              xlabel = "component", ylabel = "proportion of variance explained")
    p2 = plot(cumsum(vars),
              label = nothing, xlabel = "component",
              ylabel = "cumulative prop. of variance explained")
    plot(p1, p2, layout = (1, 2), size = (700, 400))
end

# ╔═╡ ec9179e4-eb5e-49d1-853e-f6cdfcc7bfca
md"We see that to have a proportion of variance explained near 1, we need approximatively 300 components. We decide to denoise with a proportion of variance explained of 0.99"

# ╔═╡ f8d7c484-580f-44ce-a8a7-47a14a999f0b
begin
	Mach_denoise = fit!(machine(PCA(pratio = 0.99), Standardized_Data))
	report(Mach_denoise)
	Data_denoise = MLJ.transform(Mach_denoise, Standardized_Data)
	Final_Test_Data_denoise=MLJ.transform(Mach_denoise, Final_Test_Data)
end;

# ╔═╡ 1a7ceb08-27b9-4d22-9be0-2239c0cb1007
md"
In order to see if we kept enough data and that we did not lost too much information, we can also plot them.
"

# ╔═╡ 45844c76-4e02-4851-bb4b-9a0eb625d0ed
begin
	Data_ = MLJ.transform(fit!(machine(PCA(maxoutdim = 2), Data_denoise)), Data_denoise)
	scatter(Data_.x1, Data_.x2,
					 legend = false, c = Int.(int(Data_precipitation)),
					 title = "denoised data", xlabel = "x1", ylabel = "x2")
end

# ╔═╡ e956d11f-c47a-43bb-9700-385650a4c525
begin
	Random.seed!(2);
	tsne_proj_ = tsne(Array(Data_denoise), 2, 50, 1000, 20)
	scatter(tsne_proj_[:, 1], tsne_proj_[:, 2],
	        c = Int.(int(Data_precipitation)),
			title = "denoised data",
	        xlabel = "TSne 1", ylabel = "TSne 2",
	        legend = false)
end

# ╔═╡ 22c79b71-74fd-4930-9ec8-1429391404e3
md"The plots made with PCA are similar to the standardized one, the denoising did not denaturate the data set. After the denoising, we can still observe that we might have two clusters in our data."

# ╔═╡ f305181f-bd05-4e1f-b3f4-ffa0b19bd51d
md"
# L1 & L2  regularization on the Logistic Classifier (Elastic Net)
"

# ╔═╡ b845b2ba-0491-4083-a68c-ce24cd226680
begin
    model = MLCourse.LogisticClassifier(penalty = :en)
	
	
    self_tuning_model_Ridge = TunedModel(model = model,
                                   tuning =  Grid(goal = 100),
                                   resampling = CV(nfolds = 10, rng = 751),
                                   range = [range(model, :(lambda),
                                                  lower = 100, upper = 120),
											range(model, :(gamma),
                                                  lower = 0.4, upper = 0.5)],
                                   measure = auc,
									repeats=2)
    self_tuning_Ridge = machine(self_tuning_model_Ridge,
                               Standardized_Data,
                               Data_precipitation) |> fit!
end;

# ╔═╡ ee40cd0f-7806-4dc7-805c-78256457da25
begin
	Ridge_predic=predict_mode(self_tuning_Ridge, Standardized_Data)
	ridge_report=report(self_tuning_Ridge)
end

# ╔═╡ 4d87223a-e1d2-4330-8d9b-39c0054fa996
confusion_matrix(Ridge_predic, Data_precipitation)

# ╔═╡ bdf09da5-0129-480f-b344-25c5bab8af77
begin
	ridgepredict = predict(self_tuning_Ridge, Final_Test_Data)
	ridge_predict = DataFrame(id = collect(1:1200), 
	                      precipitation_nextday = ridgepredict.prob_given_ref[2])
	Ridge_submission = CSV.write("RidgePrediction.csv", ridge_predict)
end

# ╔═╡ 50c8edf5-ae11-46bf-a73d-83b9f12ffb36
md"
# Neural Networks
"

# ╔═╡ 84263891-8765-4e71-89fb-2f8b3572c94d
begin
	NeuralModel = @pipeline(
	                       NeuralNetworkClassifier(
	                             builder = MLJFlux.Short(n_hidden = 32,
	                                                     σ = Flux.relu,
				                                         dropout = .5),
	                             optimiser = ADAM(),
	                             batch_size = 64,
								 epochs= 30),
	                       prediction_type = :probabilistic)
	
	TunedNeuralModel = TunedModel(model = NeuralModel,
								  resampling = CV(nfolds = 10, rng =3),
		                          tuning=Grid(goal = 50),
		                          range = [
									range(NeuralModel,
										     :(neural_network_classifier.lambda),
										     lower = 1e-4, upper=1e-1),
									range(NeuralModel,
										     :(neural_network_classifier.alpha),
										     lower = 0, upper=1)],
		                          measure = auc);
end

# ╔═╡ ac578d16-5b9d-4435-a01a-5433a547d6b4
begin
AUCArrays = []
	
for i in 1:10
	NeuralMach = machine(TunedNeuralModel,Data_denoise,Data_precipitation)
	fit!(NeuralMach)
	CurrentReport = report(NeuralMach)
	append!(AUCArrays,CurrentReport.best_history_entry.measurement)
	NeuralMach.model.model.neural_network_classifier.epochs +=1
end
end

# ╔═╡ 18f1f393-6e03-49a9-a300-0e5b2a20e239
plot(collect(30:39),AUCArrays)

# ╔═╡ b26d847f-d15f-4f16-923f-00e6c8aae45b
begin
	OptimalNeuralModel = @pipeline(
	                       NeuralNetworkClassifier(
	                             builder = MLJFlux.Short(n_hidden = 32,
	                                                     σ = Flux.relu,
				                                         dropout = .5),
	                             optimiser = ADAM(),
	                             batch_size = 64,
								 epochs= 33),
	                       prediction_type = :probabilistic)
	
	OptimalTunedNeuralModel = TunedModel(model = NeuralModel,
								  resampling = CV(nfolds = 10, rng =3),
		                          tuning=Grid(goal = 50),
		                          range = [
									range(NeuralModel,
										     :(neural_network_classifier.lambda),
										     lower = 1e-4, upper=1e-1),
									range(NeuralModel,
										     :(neural_network_classifier.alpha),
										     lower = 0, upper=1)],
		                          measure = auc);
end

# ╔═╡ e13278f5-576d-43e4-897a-3ffad4cf7a58
begin
OptimalNeuralMach = fit!(machine(OptimalTunedNeuralModel,Data_denoise,Data_precipitation));
end

# ╔═╡ 3bfa6b33-700f-4ec0-a463-ccbe1bd667b1
report(OptimalNeuralMach)

# ╔═╡ eee8b886-b0eb-43c5-925d-9c1b5a86fb09
confusion_matrix(predict_mode(OptimalNeuralMach, Data_denoise),
                Data_precipitation)

# ╔═╡ cceff046-bce3-4356-907d-21a1864d8a3a
Neuralprediction = predict(OptimalNeuralMach,Final_Test_Data_denoise)

# ╔═╡ 70e4bc76-3584-4ff9-a25e-7e22867ccd8f
neural_prediction = DataFrame(id = collect(1:1200), 
	                      precipitation_nextday = Neuralprediction.prob_given_ref[2])

# ╔═╡ 2c788bf3-7a33-4769-b2d1-9369d199167e
CSV.write("NeuralPrediction.csv", neural_prediction)

# ╔═╡ Cell order:
# ╠═9ff5b5bc-eb0f-456b-852e-22004fb52634
# ╠═cc54ecc2-cfcb-4096-b5c4-d58f1e91b14d
# ╠═a4d202fa-c71e-4599-ad15-2874f95ea98b
# ╠═a88c2ef8-e16c-4db0-b826-16d7ab86e0dc
# ╠═fbfd2af3-b31f-4674-9854-4f60cb10c6ae
# ╠═420cb2b8-ce8e-4b82-abc1-3f16da357982
# ╠═16be29f3-31ae-4fe9-bb4c-b4cb17d019bc
# ╠═30c875e8-ffea-4391-91e7-a6fbbc639cd8
# ╟─4d2179df-cbf9-42dd-b6b0-021a98d29259
# ╠═d67c9d59-53ff-4091-a810-13490ca7c3db
# ╠═af2619ab-b851-4f0a-9a81-03b513cbebca
# ╟─5737aa4e-dd68-449a-9909-156a2ed74519
# ╠═e24d2e2d-381a-4777-af79-49aa44d4e374
# ╟─058da5bf-a59b-4e00-9698-ff3d0852d421
# ╠═59fa7078-7b4f-4984-b676-698d6ffc496d
# ╟─1cd896e7-70e0-4178-992d-7b2e1ece1839
# ╠═136c3257-f70f-4acb-8fbb-316ea4578835
# ╟─c5984f1e-28d5-4b45-8d86-c0b0142579f8
# ╠═6ec9e33f-8e37-43b6-ad10-38d0c3f65e58
# ╠═3f3a2c4a-a6a9-4d4a-a8f7-2d9fe9aa2492
# ╟─298772c2-01e3-4ea7-854b-1afc4fa7a054
# ╠═f75efce8-2a3c-4f50-b475-23b72fefd654
# ╟─adb61dc7-ddfd-4c39-8924-264b992461fe
# ╠═cf6d997f-5aa1-4b5e-bedd-581c27a71d58
# ╠═526f11c6-045b-4fbe-8cb7-61d7c4e73ebd
# ╟─9955afa5-413d-4de5-9d25-d3f04710764c
# ╠═d30f7886-f509-48ca-93d5-447a56e75b73
# ╟─d86955c2-e84b-46f6-9df1-93fd5602ccef
# ╠═c376351c-8373-46f9-8180-e1510060f0f8
# ╟─521c187d-f983-4d31-8b1c-c0e2f88b9471
# ╟─67e5045c-6711-463a-8670-0d70e318ffc7
# ╟─f706d246-7f28-4018-9240-d00a2cfeac7f
# ╟─004aa8fb-ea81-4458-80af-f6c7d182f4fc
# ╟─30b08bdc-09da-4780-aae8-d1624559a6a4
# ╟─4cc568f3-d1a1-45cd-a793-e4e518d967d8
# ╟─25579abf-ab22-44dd-b052-cf5b07d1a72c
# ╟─114e4c22-b71f-4208-b543-a2b259941a38
# ╟─74d061c3-c0e4-43fa-a9dd-07f59c24a2b3
# ╟─fa5b2000-4541-4f12-89ae-6471bed07883
# ╟─3e0c41a1-91ac-4d3a-a907-9790507e63e1
# ╟─26843343-cf84-4a58-a158-731bdb27ba40
# ╟─61b96210-d752-4980-b208-325df173a7f1
# ╠═14d9e84f-f146-45ff-9cd0-81048b415db6
# ╟─3777aa73-65a0-42a4-a9c4-833d71e637cb
# ╟─247918e3-afc6-4a12-b5ff-24d52914aa44
# ╟─894db97e-07c4-4da0-a443-a4b6d3effb85
# ╟─ec9179e4-eb5e-49d1-853e-f6cdfcc7bfca
# ╠═f8d7c484-580f-44ce-a8a7-47a14a999f0b
# ╟─1a7ceb08-27b9-4d22-9be0-2239c0cb1007
# ╟─45844c76-4e02-4851-bb4b-9a0eb625d0ed
# ╠═e956d11f-c47a-43bb-9700-385650a4c525
# ╟─22c79b71-74fd-4930-9ec8-1429391404e3
# ╟─f305181f-bd05-4e1f-b3f4-ffa0b19bd51d
# ╠═b845b2ba-0491-4083-a68c-ce24cd226680
# ╠═ee40cd0f-7806-4dc7-805c-78256457da25
# ╠═4d87223a-e1d2-4330-8d9b-39c0054fa996
# ╠═bdf09da5-0129-480f-b344-25c5bab8af77
# ╟─50c8edf5-ae11-46bf-a73d-83b9f12ffb36
# ╠═84263891-8765-4e71-89fb-2f8b3572c94d
# ╠═ac578d16-5b9d-4435-a01a-5433a547d6b4
# ╠═18f1f393-6e03-49a9-a300-0e5b2a20e239
# ╠═b26d847f-d15f-4f16-923f-00e6c8aae45b
# ╠═e13278f5-576d-43e4-897a-3ffad4cf7a58
# ╠═3bfa6b33-700f-4ec0-a463-ccbe1bd667b1
# ╠═eee8b886-b0eb-43c5-925d-9c1b5a86fb09
# ╠═cceff046-bce3-4356-907d-21a1864d8a3a
# ╠═70e4bc76-3584-4ff9-a25e-7e22867ccd8f
# ╠═2c788bf3-7a33-4769-b2d1-9369d199167e
