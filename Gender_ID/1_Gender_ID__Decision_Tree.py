from sklearn import tree #A desicion tree is like a flowchart that store data

#[Height, Weight, shoe size]
X = [[181,80,44],[177,70,43],[160,60,38],[154,54,37],[166,65,40],[190,90,47],
     [175,64,39],[177,70,40],[159,55,37],[171,75,42],[181,85,43]]
m = "male"
f = "female"
Y = [m,f,f,f,m,m,m,f,m,f,m]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

prediction = clf.predict([[190,70,43]])

print(prediction)

"""
Challenge:
    1. Use any 3 Scikit-Learn Models on this dataset.
    2. Compare results.
    3. Print the best one.
    4. Use a larger data set.
"""
