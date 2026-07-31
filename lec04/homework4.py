
def next_birthday(date, birthdays):
    '''
    Find the next birthday after the given date.

    @param:
    date - a tuple of two integers specifying (month, day)
    birthdays - a dict mapping from date tuples to lists of names, for example,
      birthdays[(1,10)] = list of all people with birthdays on January 10.

    @return:
    birthday - the next day, after given date, on which somebody has a birthday
    list_of_names - list of all people with birthdays on that date
    '''
    # Find all birthdays after the given date
    future_birthdays = [b for b in birthdays.keys() if b > date]
    
    if future_birthdays:
        # Find the earliest birthday after the given date
        birthday = min(future_birthdays)
    else:
        # Wrap around: find the earliest birthday in the year
        birthday = min(birthdays.keys())
    
    list_of_names = birthdays[birthday]
    return birthday, list_of_names
    
