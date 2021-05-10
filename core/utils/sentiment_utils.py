def getSentiment(self, polarity):
    """
        |-- polarity <= -0.3 : negative
        |-- polarity < 0.3 : neutral
        |-- polarity >= 0.3 : positive
    """

    if polarity <= -0.3:
        return -1
    elif polarity <= 0.3:
        return 0
    else:
        return 1
