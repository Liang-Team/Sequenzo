from sequenzo.clustering import kmedoid_cpp

class KMedoids:
    def __init__(self,nelements,diss,centroids,npass,weights):
        self.model = kmedoid_cpp.KMedoid(nelements, diss, centroids, npass, weights)
    def runclusterloop(self):
        return self.model.runclusterloop()