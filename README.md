# Sensorless Dust Detection for Optimal Solar Panel Cleaning

ABSTRACT

Investment in solar photovoltaics (PV) systems has increased to meet global green energy targets. However, dust accumulation on PVs creates a significant reduction in output power and increases cleaning costs. At a global PV capacity above 500 GW, billions of gallons of water are used annually for cleaning purposes. This is problematic for desert areas where water is scarce, especially because the abundance of sunlight in these regions makes them favorable for solar power production. Thus, optimizing PV cleaning schedules is crucial to make solar energy more sustainable. Past research has used devices including cameras and sensors for dust detection; however, this created additional maintenance costs and complications. Other proposed algorithms for cleaning schedules have also been done on a yearly basis, making them less specialized. Therefore, this study proposed an algorithm that used inputs that are easily accessible online in order to determine whether cleaning is necessary every week, making it more specific to each case. A feed-forward neural network was used to predict output power based on environmental factors. The network’s performance was enhanced through bootstrapping and extensive testing to ensure compatibility with the data. The algorithm is validated through a simulation using the above model and the environmental data. Results showed that the maximum output power increase is 9.577%, which accompanies a 95.83% decrease in cleaning events using this algorithm. This indicates that there is great potential to improve the economic viability and resource efficiency of PV systems without needing to create drastic technological advancements in solar panel material.

This project was submitted to the Synopsys Science Fair.
This is the [poster](https://docs.google.com/presentation/d/1ASyeUAI812egtVX5WX9Ho97XWOXz4f_9/edit?usp=sharing&ouid=114194103913446610142&rtpof=true&sd=true).

In this repository, I have laid out the code for this project, as well as the experimentation done along the way. Here is the [project notebook ](https://docs.google.com/document/d/1MjkvpkYqd8QFvVIVOXfncDkNxMwyUVdyQC1m9oImqd8/edit?usp=sharing)with more detailed information.
The final code mainly involves run.py, which is the implementation of the core optimized cleaning algorithm.
The main helper files are accessData.py and OutputNumberPredictionModel.py. The latter file uses a feed-forward neural network to predict the output power production of a solar panel based on the environmental data.

This code is used along with this [dataset](https://datahub.duramat.org/dataset/pvdaq-time-series-with-soiling-signal/resource/d2c3fcf4-4f5f-47ad-8743-fc29f1356835).
