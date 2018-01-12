BigData Project-Readme

1) Data Downloaded from : https://cosic.esat.kuleuven.be/fingerprintability/  -- This is the sample data link given by the authors of paper
2) Ran pig script for preprocessing : uploaded in drive(add_filename.pig) : pig -x local pig_file_name - command to run
3) Ran Naivebayes classifier : output screenshot(Naive Bayes.png) and code is uploaded in drive(project_NB.py)
4) Ran Logistic regression code : output not generated as its giving timeout exception but if ran on one file the attack prooves to make a strong case to be viable alternative, code uploaded in drive(project.py)
5) Ran Kmeans || : output wasn't much satistfying except bolstering our intution that immune no. of web-services are significantly more than vulnerable no. of web-services.
6) Ran perceptron model of Machine Learning : output gives high accuracy in prediction thus the attack prooves to make a strong case to be viable alternative.
7) We also have one more data which we extracted with traverse_shell.sh(tshark code) which after converting to csv files is only 320Mb and 14gb of logfiles. 
we did not know how to interpret logfiles because of which we did not use it.
