### GET DUMMIES / 1 hot encoding ###

wbr.weathersit.hist()

dummies = pd.get_dummies(wbr.weathersit)
colnames = { 1:'sunny', 2:'cloudy', 3:'rainy'} #This is a dictionary
dummies.rename(columns = colnames, inplace = True) #Rename column label
wbr = pd.concat([wbr,dummies],axis=1) #Add new columns