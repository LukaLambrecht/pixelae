import os
import sys
import json
import numpy as np


def parse_oms_timestamp(oms_timestamp):
        part1, part2 = oms_timestamp.split('T')
        parts1 = part1.split('-')
        year = int(parts1[0])
        month = int(parts1[1])
        day = int(parts1[2])
        parts2 = part2.strip('Z').split(':')
        hour = int(parts2[0])
        minute = int(parts2[1])
        second = int(parts2[2])
        timestamp = (
          year * 10000000000
          + month * 100000000
          + day * 1000000
          + hour * 10000
          + minute * 100
          + second
        )
        return timestamp


class LumisectionTimeParser(object):

    def __init__(self, jsonfile):
        
        # read json file and parse attributes
        with open(jsonfile, 'r') as f:
            info = json.load(f)
        self.runs = np.array(info['run_number'])
        self.lumis = np.array(info['lumisection_number'])
        self.start_times = np.array([parse_oms_timestamp(t) for t in info['start_time']])
        self.end_times = np.array([parse_oms_timestamp(t) for t in info['end_time']])

        # do sorting (todo)
        start_times_is_sorted = np.all(self.start_times[:-1] <= self.start_times[1:])
        end_times_is_sorted = np.all(self.end_times[:-1] <= self.end_times[1:])
        if (not start_times_is_sorted) or (not end_times_is_sorted):
            msg = 'Provided lumisection appear to be not time-ordered,'
            msg += ' dealing with this is not yet implemented.'
            raise Exception(msg)

    def get_time(self, run, lumi):
        ids = np.nonzero((self.runs==run) & (self.lumis==lumi))[0]
        if len(ids)!=1:
            msg = f'Run {run}, LS {lumi} not found.'
            raise Exception(msg)
        idx = ids[0]
        return (self.start_times[idx], self.end_times[idx])

    def get_lumi(self, timestamp):
        start_idx = np.searchsorted(self.start_times, timestamp, side='right')
        if start_idx == 0:
            msg = f'Timestamp {timestamp} is earlier than minimum ({self.start_times[0]})'
            raise Exception(msg)
        start_idx = start_idx - 1
        end_idx = np.searchsorted(self.end_times, timestamp, side='right')
        if end_idx == len(self.end_times):
            msg = f'Timestamp {timestamp} is later than maximum ({self.end_times[-1]})'
            raise Exception(msg)
        if start_idx != end_idx:
            msg = f'Timestamp {timestamp} seems to fall outside of any lumisection.'
            raise Exception(msg)
        run = self.runs[start_idx]
        lumi = self.lumis[start_idx]
        return (run, lumi)
