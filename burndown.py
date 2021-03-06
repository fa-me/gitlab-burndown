import sys
import gitlab
import collections
import datetime
import dateutil.parser
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.interpolate as interpolate
import scipy.signal as signal

import pandas as pd


def get_issues(gitlab_url, gitlab_secret, project, since):
    gl = gitlab.Gitlab(gitlab_url, gitlab_secret)
    proj = gl.projects.get(project)

    done = False
    page = 1

    all_issues = []

    while not done:
        issues = proj.issues.list(order_by='created_at',
                                  sort='desc',
                                  page=page,
                                  per_page=20)
        if len(issues) == 0:
            break
        page += 1

        all_issues += issues

    return all_issues


milestone_lookup = dict()


def issue_to_dict(issue):
    # open time
    open_time = dateutil.parser.parse(issue.created_at)
    close_time_raw = issue.attributes['closed_at']

    if close_time_raw is not None:
        close_time = dateutil.parser.parse(close_time_raw)
    else:
        close_time = None

    # milestone
    if issue.milestone is not None:
        milestone_id = issue.milestone['iid']

        if issue.milestone['start_date'] is None:
            ms_start_date = None
        else:
            ms_start_date = dateutil.parser.parse(issue.milestone['start_date'])

        milestone_lookup.update({milestone_id: {
            'title': issue.milestone['title'],
            'start_date': ms_start_date,
            'due_date': dateutil.parser.parse(issue.milestone['due_date'])
        }})
    else:
        milestone_id = None

    return dict({
        'iid': issue.get_id(),
        'open_time': open_time,
        'close_time': close_time,
        'milestone_id': milestone_id
    })


def issues_to_dataframe(issues):
    return pd.DataFrame(map(issue_to_dict, issues))


def get_timestamps(from_, to_, freq='D'):
    return pd.date_range(from_, to_, freq=freq).tolist()


def get_bracket_counter(df, col):
    def in_time_bracket(lower, upper):
        return np.count_nonzero(np.logical_and(df[col] > lower,
                                               df[col] <= upper))
    return in_time_bracket


def get_start_due_date(df):
    milestone_ids = set(df.milestone_id)

    milestone_id = milestone_ids.pop()
    milestone = milestone_lookup[milestone_id]
    start_date = milestone['start_date']
    due_date = milestone['due_date']
    return start_date, due_date


def accumulated_number_of_items(df, freq='D'):
    earliest = df.open_time.min()
    tz = earliest.tz

    now = pd.Timestamp.now(tz=tz)
    latest = max(df.open_time.max(),
                 df.close_time.max(),
                 now)

    timestamps = get_timestamps(earliest, latest, freq)

    timestamps.append(latest)  # assert latest is explicitly added

    count_opened = get_bracket_counter(df, 'open_time')
    count_closed = get_bracket_counter(df, 'close_time')

    timestamp_brackets = list(zip(timestamps[:-1], timestamps[1:]))

    opened = [count_opened(a, b)
              for a, b in timestamp_brackets]
    closed = [count_closed(a, b)
              for a, b in timestamp_brackets]

    return timestamps[1:], opened, closed


def plot_data(df, freq='D', title=None):
    t, opened, closed = accumulated_number_of_items(df, freq)

    opened_cum = np.cumsum(opened)

    plt.figure(figsize=(12,4))

    start_date, due_date = get_start_due_date(df)
    if start_date is None:
        start_date = min(t)

    plt.plot([start_date, due_date], [0, opened_cum[-1]], '--', color='0.5')

    plt.fill_between(t, opened_cum, color='0.9')
    plt.fill_between(t, np.cumsum(closed))

    if title is not None:
        plt.title(title)

    plt.xlim(min(t), max(t), due_date)
    plt.ylim(0, max(opened_cum))
    plt.tight_layout()
    plt.show()


def main(gitlab_url=None, gitlab_secret=None, project=None, since=None):

    issues = get_issues(gitlab_url, gitlab_secret, project, since)
    data = issues_to_dataframe(issues)

    for milestone_id, subdata in data.groupby('milestone_id'):
        milestone_name = milestone_lookup[milestone_id]['title']
        plot_data(subdata, freq='H', title=milestone_name)


if __name__ == "__main__":
    pass
