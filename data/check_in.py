class CheckIn:
    id = None
    child_name = None
    organisation_id = None
    attendance_date = None
    rec_time = None
    photo_url = None

    def __init__(self, check_in_dict):
        self.id = check_in_dict["checkin_id"]
        self.child_name = check_in_dict["child_name"]
        self.organisation_id = check_in_dict["organisation_id"]
        self.attendance_date = check_in_dict["attendance_date"]
        self.rec_time = check_in_dict["rec_time"]
        self.thumbnail = check_in_dict["thumbnail"]
        self.photo_url = check_in_dict["photo"]
