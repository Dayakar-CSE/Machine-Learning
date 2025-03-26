"""
    # Applying Bayes' Theorem to calculate P(Absent | Friday)

# Given probabilities
P_Friday_and_Absent = 0.03  # Probability that it's Friday and the student is absent
P_Friday = 0.20             # Probability that it's Friday

# Bayes' Rule: P(Absent | Friday) = P(Friday and Absent) / P(Friday)
P_Absent_given_Friday = P_Friday_and_Absent / P_Friday

# Display the result
print(f"Probability that a student is absent given that today is Friday: {P_Absent_given_Friday:.2%}")
"""





PFIA=float(input("Enter probability that it is Friday and that a student is absent="))
PF=float(input(" probability that it is Friday="))
PABF=PFIA / PF
print("probability that a student is absent given that today is Friday using conditional probabilities=",PABF)
