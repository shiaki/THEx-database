
#
def save_snapshot(snapshots, rec):
    ''' Keep a snapshot of original record (in 'snap' array). '''
    snapshots.update_one({'_id': rec['_id']}, {'$addToSet': {'snap': rec}})
    snapshots.update_one({'_id': rec['_id']}, {'last_update': date_stamp()})

def into_boneyard(events, boneyard, rec):
    ''' Keep a snapshot of original record. '''
    boneyard.insert_one({**rec, **{'last_update': date_stamp()}})
    events.delete_one({'_id': rec['_id']})
