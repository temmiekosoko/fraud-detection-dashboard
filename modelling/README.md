# Data Scientist Position: *Take Home Assignment*

This assignment is designed to give us a sense of your comfort level working with programming languages, concepts, and
methods that are key to your success as a Data Scientist at Appriss.

We are not looking for perfect or exhaustive solutions. We
primarily want to see that you know what you are doing and that you are applying concepts and methods appropriately. If
you think certain tasks are unclear, simply use your best guess about what is meant (you can write comments in your
code explaining your choices if you think this is necessary).

Once you have completed the assignment, please make a pull request with your changes.

## Assignment Description

*Verify* is an Appriss machine learning solution helping retailers detect and prevent return fraud. Return fraud is the
abuse of a retailer's return policy by a customer. For instance, a person may repeatedly buy new shoes from a retailer,
wear them for a few weeks and then return them for a refund. The *Verify* model is designed to identify such customers,
so that the retailer can deny their return.

In this assignment, you are tasked with building a proof-of-concept for improving the *Verify* model by adding a new
feature, creating a new data preprocessing method, training a model with the new feature and comparing it to the
existing model.

### Data Description

As your data source you will use the sample data provided in the sqlite database *sample.db* in this repo. The database
consists of three tables.

**Table Name:** transactions \
**Table Description:** This table contains a sample of purchase and return transactions made across different retailers.
A transaction can consist of several items purchased or returned. For each item purchases or returned, there is a
separate row. A row is thus uniquely identified by the tuple (retailer_id, transaction_id, item_id, sale).

| Name                | Description                                        |
|---------------------|----------------------------------------------------|
| retailer_id         | Retailer identifier                                |
| transaction_id      | Transaction identifier                             |
| transaction_time    | Time of the transaction                            |
| item_id             | Item identifier                                    |
| sale                | Is the item purchased ('Y') or returned ('N')?     |
| orig_transaction_id | The transaction identifier of the initial purchase |

**Table Name:** customers \
**Table Description:** This table associates a transaction with a customer.

| Name           | Description                                                     |
|----------------|-----------------------------------------------------------------|
| retailer_id    | Retailer identifier (matches retailer_id in transactions)       |
| transaction_id | Transaction identifier (matches transaction_id in transactions) |
| customer_id    | Customer identifier                                             |

**Table Name:** model_data \
**Table Description:** This table contains the target and existing features for model training. For each return in the
transactions table there is an associated entry in this table, specifying if the return was fraudulent, the prediction
of the current model, and several already existing features that might be helpful for model training later on.

| Name           | Description                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------------------|
| retailer_id    | Retailer identifier (matches retailer_id in transactions)                                                   |
| transaction_id | Transaction identifier (matches transaction_id in transactions)                                             |
| item_id        | Item identifier (matches item_id in transactions)                                                           |
| return_fraud   | True label. Was the return fraudulent (1) or not (0)                                                        |
| model_pred     | Prediction of the current model                                                                             |
| return_rate    | The customer's (dollar amount) return rate over the last 365 days since the time of the transaction         |
| num_pur        | The number of items the customer has purchased in the last 365 days since the time of the transaction       |
| num_ret        | The number of items the customer has returned in the last 365 days since the time of the transaction        |
| amt_ret        | The total dollar amount of returns the customer made in the last 365 days since the time of the transaction |
| amount         | The dollar amount of the current return                                                                     |

### Tasks

#### 1. Feature Engineering

A key indicator for fraudulent activity is the number of returns a customer has made in the past year as captured by the
`num_ret` variable in the *model_data* table. However, it occasionally happens that there are errors at the register and
the cashier immediately voids the purchase by making a return for the customer. The current variable counts such returns
just like any other, but it might improve the model to exclude them because the return was not done by the customer.
As such transactions usually happen in quick succession, a simple fix is to not count returns made within 5 minutes of
the original purchase.

- Write the appropriate SQL code for calculating a new feature called `num_true_ret` counting all returned items within
  365 days of the current return but only if a return was made at least 5 minutes after the time of the purchase.
- In `create_dataset.py`, write a script that executes your SQL code and creates a new table in the database
  called *model_data_new* that consists of the current *model_data* table appended with the new `num_true_ret` column.

**Hint:** Timestamp manipulation capabilities are somewhat limited in SQLite. However, you can make use of the
`julianday()` function which returns the fractional number of days since noon in Greenwich on November 24, 4714 B.C..
For instance, `select (julianday('2022-01-02') - julianday('2022-01-01')) * 24 * 60;` returns the time in minutes
between January 1, 2022 and January 2, 2022.

**Example:** For a single customer's transaction history, your new feature should look like this:

| retailer_id | transaction_id | transaction_time    | item_id | sale | orig_transaction_id | num_ret | num_true_ret |
|-------------|----------------|---------------------|---------|------|---------------------|---------|--------------|
| C           | 53098          | 2020-03-02 08:13:16 | 10076   | N    |                     | 1       | 1            |
| C           | 116571         | 2021-05-19 13:45:52 | 15353   | Y    |                     | 0       | 0            |
| C           | 116571         | 2021-05-19 13:45:52 | 20993   | Y    |                     | 0       | 0            |
| C           | 105594         | 2021-05-19 13:49:12 | 20993   | N    | 116571              | 1       | 0            |
| C           | 49093          | 2021-06-10 12:31:42 | 19019   | Y    |                     | 1       | 0            |
| C           | 120126         | 2021-06-10 14:18:14 | 19019   | N    | 49093               | 2       | 1            |

#### 2. Data ETL

To train a model with the new feature, the dataset must be prepared in SQL and loaded into Python. To improve
reusability, the code should follow OOP principles.

- Create a Python script called `main.py` that serves as the entry point to running this test. 
  - It must accept a command line argument that indicates the type of database. This test uses SQLite but assume that 
    it will include PostgreSQL in the future and maybe others as well.
- Extend the code in `data_etl.py` by subclassing the abstract base class `DataLoader`.
    - The new class should be initialized with a sqlite connection object and implement a load method
    - The load method should take the name of the target variable and the names of desired features as input and load
      the appropriate data from the database into a pandas dataframe.

#### 3. Model Training

- In `train_model.py`, write a script that
    - loads the data from the sqlite database and prepares the model dataset using the DataLoader class you just created
    - preprocesses the dataset in whatever way you see fit
    - trains a model to predict the return_fraud label (you may choose any model you like)
    - compares your model to the existing model across one or two metrics (you can choose what is appropriate) and
      prints out results (simple print statements are sufficient)

#### 4. Unit Testing

- Write unit tests for the included ModelPrediction class's methods.
- Note that this task is independent of the preceding steps. Do not include ModelPrediction in `train_model.py`.
- pytest and unittest are both acceptable frameworks.
- Include a comment indicating how to run the unit test from the command line.
- Dataclasses like ModelPrediction have autogenerated __init__ functions. E.g.:
	- my_prediction = ModelPrediction('C', 1001, 1003, True)
	- my_prediction = ModelPrediction(retailer_id = 'C', transaction_id = 1001, item_id = 1003, return_fraud = True)
	- my_prediction = ModelPrediction('C', 1001, item_id = 1003, return_fraud = True)
