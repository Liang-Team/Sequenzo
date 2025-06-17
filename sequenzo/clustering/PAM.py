from sequenzo.clustering import pam_cpp

class PAM:
    def __init__(self,nelements,diss,centroids,npass,weights):
        self.model = pam_cpp.PAM(nelements, diss, centroids, npass, weights)
    def runclusterloop(self):
        return self.model.runclusterloop()