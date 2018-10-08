# twitter-case

Last sprint to-do's:
  - Correlation between trump vote and trump sentiment at the county level
  - build and illustrate topic model
  - Build model to predict results of elections (regression)
  - Run topic model for two selected states: preferably one with the highest Trump-vote and one with highest Hillary-vote

For the ones new to Github (like me) this short video was pretty helpful in explaining how it works: https://www.youtube.com/watch?v=0fKg7e37bQE

Lessons learnt:
  - Reduce date before doing calculations
  - Make sure you know the ins and outs of your sentiment library (vader takes capitals into account and we used w.lower() and removed special characters)
  - More focus on replicability (streamlining of workflow and methods)
  - A better approach for the topic model might be to take observations that are above/below 2 S.D. in sentiment related to the candidate and model their tweets... 
  
  Note to self for next case:
  - See if we can reduce the scale of PERMA
