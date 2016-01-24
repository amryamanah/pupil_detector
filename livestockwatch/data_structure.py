# -- coding: utf-8 --
__author__ = 'amryfitra'


class EyeCheckerFeatures(object):
    # Shape related features for checking existance of eye in a frame
    def __init__(self):
        self.contour_area = None
        self.aspect_ratio = None
        self.bounding_box_area = None
        self.convex_hull = None
        self.extent = None
        self.solidity = None


class RoiFeatures(object):
    # Shape related features for checking ROI of eye in a frame with eye
    def __init__(self):
        self.roi_contour_area = None
        self.roi_aspect_ratio = None
        self.roi_bounding_box_area = None
        self.roi_convex_hull = None
        self.roi_extent = None
        self.roi_solidity = None


class PLRFeatures(object):
    # ROI Features for getting PLR related features
    def __init__(self):
        self.area = None
        self.eccentricity = None


class SurRefFeatures(object):
    # ROI Features for getting Surface reflectance related features
    def __init__(self):
        #### RGB colorspace
        self.red_mean = None
        self.green_mean = None
        self.blue_mean = None

        self.red_std = None
        self.green_std = None
        self.blue_std = None

        #### L*a*b* colorspace
        self.lightness_mean = None
        self.a_mean = None
        self.b_mean = None

        self.lightness_std = None
        self.a_std = None
        self.b_std = None

        #### HSV colorspace
        self.hue_mean = None
        self.val_mean = None
        self.sat_mean = None

        self.hue_std = None
        self.val_std = None
        self.sat_std = None


class FrameData(object):
    def __init__(self):
        self.filename = None
        self.imgtype = None
        self.filetype = None
        self.timestamp = None
        self.roi_flag = False
        self.frame_number = None
        self.eye_checker_feature = EyeCheckerFeatures()
        self.roi_feature = RoiFeatures()
        self.sur_ref_feature = SurRefFeatures()
        self.plr_feature = PLRFeatures()


class CaptureSession(object):
    def __init__(self):
        self.cs_name = None
        self.folderpath = None
        self.totd = None
        self.datetime_taken = None
        self.stall_name = None
        self.cattle_id = None
        self.plr = None
        self.ca = None
        self.reflectance_intensity = None
        self.pupil_color_intensity = None
        # List of FrameData
        self.lst_frame = []

    def get_pl_frame(self):
        return [frame for frame in self.lst_frame if frame.imgtype == "PL"]

    def get_nopl_frame(self):
        return [frame for frame in self.lst_frame if frame.imgtype == "NOPL"]

    def get_frame_with_eye(self):
        return [frame for frame in self.lst_frame if frame.roi_flag]

    def get_frame_without_eye(self):
        return [frame for frame in self.lst_frame if not frame.roi_flag]

