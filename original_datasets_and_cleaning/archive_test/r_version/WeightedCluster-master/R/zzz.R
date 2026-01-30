
.onAttach <- function(libname, pkgname){
	suppressWarnings(descr <- utils::packageDescription("WeightedCluster"))
	if(utils::packageVersion("WeightedCluster")$minor %% 2 == 0) {
		state <- "stable"
	}
	else {
		state <- "development"
	}
	if(!is.null(descr$Built)){
		builtDate <- paste(" (Built: ", strsplit(strsplit(descr$Built, ";")[[1]][3], " ")[[1]][2], ")", sep="")
	}else{
		builtDate <- ""
	}
	packageStartupMessage("This is WeightedCluster ", state, " version ", descr$Version, builtDate)
	packageStartupMessage('\nTo access available manuals and short tutorials, please run:')
	packageStartupMessage('   vignette("WeightedCluster") ## For the complete manual in English')
	packageStartupMessage('   vignette(package="WeightedCluster") ## To list available documentation')
	#packageStartupMessage('   vignette("WeightedClusterPreview") ## Short preview in English')
	packageStartupMessage("\nTo cite WeightedCluster in publications please use or references in the help pages:")
	x <- "Studer, Matthias (2013). WeightedCluster Library Manual: A practical guide to creating typologies of trajectories in the social sciences with R. LIVES Working Papers, 24. doi: 10.12682/lives.2296-1658.2013.24"
	sapply(strwrap(x, exdent=3), packageStartupMessage)
	invisible()
	
}
