def get_dataset_labels(df, columns = ['text','hard_label','soft_label_0','soft_label_1', 'disagreement']):
  """
  df: dataframe to elaborate
  colums: list of output columns
  ______________________________
  Extract two columns from the soft-label column to represent disagreement on the positive and negative label.
  Add a "disagreemen" column with boolean values (1 for agreement, 0 for disagreement)
  Rename the column "text" in "original text" to distiguish with the token-column "text"
  """
  df['soft_label_1']= df['soft_label'].apply(lambda x: x['1'])
  df['soft_label_0']= df['soft_label'].apply(lambda x: x['0'])
  df['disagreement'] = df['soft_label_0'].apply(lambda x : int(x==0 or x==1))
  df.rename({'text': 'original_text'}, axis=1, inplace=True)
  return df[columns]

def soft_labels(predictions):
  if 'YES' in predictions:
    return predictions.count('YES')/6
  else:
    return 0

def most_frequent(List): #hard label
  counter = 0
  value = List[0]

  for i in List:
      curr_frequency = List.count(i)
      if(curr_frequency> counter):
          counter = curr_frequency
          value = i
  # if 50% agreement, than positive label
  if counter == 3:
    value = 'YES'
  return value, counter

