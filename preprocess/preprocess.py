class PreProcessor(object):

    def __init__(self):
        self.all_tri_type = ["Regulation", "Cell_proliferation", "Gene_expression", "Binding", "Positive_regulation",
                             "Transcription", "Dephosphorylation", "Development", "Blood_vessel_development",
                             "Catabolism", "Negative_regulation", "Remodeling", "Breakdown", "Localization",
                             "Synthesis", "Death", "Planned_process", "Growth", "Phosphorylation"]
        self.dir = "../resource/"
        self.output_dir = "../example/"
