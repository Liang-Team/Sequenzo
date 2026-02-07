## weighted boxplots
## Stuff from ENmisc (by Erich Neuwirth <erich.neuwirth at univie.ac.at>)
##  included here to avoid issues with dependence on ENmisc

wtd.boxplot.tmr <-
function(x, weights=NULL, ..., range = 1.5, width = NULL, varwidth = FALSE,
         notch = FALSE, outline = TRUE, names, plot = TRUE,
         border = par("fg"), col = "lightgrey", log = "",
         pars = list(boxwex = 0.8, staplewex = 0.5, outwex = 0.5),
         horizontal = FALSE, add = FALSE, at = NULL,
         frame.plot = FALSE)
{
    args <- list(x, ...)
    namedargs <-
        if(!is.null(attributes(args)$names))
            attributes(args)$names != ""
        else
            rep(FALSE, length.out = length(args))
    pars <- c(args[namedargs], pars)
    groups <- if(is.list(x)) x else args[!namedargs]
    if (!is.null(weights)){
    if(!is.list(weights)) weights<-list(weights)
        datasize<-sapply(groups,length)
        wtsize<-sapply(weights,length)
        if (length(datasize)!=length(wtsize))
          stop("number of groups for data and weights are different")
        if (any(datasize != wtsize))
          stop("group sizes for data and weights are different")
        groupwts<-weights
    }
    else groupwts<-NULL
    if(0 == (n <- length(groups)))
        stop("invalid first argument")
    if(length(class(groups)))
        groups <- unclass(groups)
    if(!missing(names))
        attr(groups, "names") <- names
    else {
        if(is.null(attr(groups, "names")))
            attr(groups, "names") <- 1:n
        names <- attr(groups, "names")
    }
    for(i in 1:n) {
        if(is.null(groupwts[[i]]))
            groups[i] <- list(wtd.boxplot.stats.tmr(groups[[i]],
                              weights=NULL,
                              coef=range)) # do.conf=notch)
        else
            groups[i] <- list(wtd.boxplot.stats.tmr(groups[[i]],
                              weights=groupwts[[i]],
                              coef=range)) # do.conf=notch)
    }
    stats <- matrix(0,nrow=5,ncol=n)
    conf  <- matrix(0,nrow=2,ncol=n)
    ng <- out <- group <- numeric(0)
    ct <- 1
    for(i in groups) {
        stats[,ct] <- i$stats
        conf [,ct] <- i$conf
        ng <- c(ng, i$n)
        if((lo <- length(i$out))) {
            out   <- c(out,i$out)
            group <- c(group, rep.int(ct, lo))
        }
        ct <- ct+1
    }
    z <- list(stats = stats, n = ng, conf = conf, out = out, group = group,
              names = names)
    if(plot) {
        bxp(z, width, varwidth = varwidth, notch = notch, log = log,
#            border = border, col = col, pars = pars,
            border = border, boxfill = col, pars = pars,
            outline = outline, horizontal = horizontal, add = add, at = at,
            frame.plot = frame.plot)
    invisible(z)
    }
    else z
}

wtd.boxplot.stats.tmr <-
function(x, weights=NULL, coef = 1.5, do.conf=TRUE, do.out=TRUE)
{

    nna <- !is.na(x)
    n <- sum(nna)                       # including +/- Inf
#   stats <- stats::fivenum(x, weights=weights, na.rm = TRUE) # is the new call
# the previous lines needs to be uncommented fot inclusion in the R distribution
# and the next line has to be deleted
    if (length(x) > 1)
        stats <- wtd.fivenum.tmr(x, weights=weights, na.rm = TRUE) # is the call for the test version
    else
        stats <- rep(x,5)
    iqr <- diff(stats[c(2, 4)])
    if(coef < 0) stop("'coef' must not be negative")
    if(coef == 0)
        do.out <- FALSE
    else {                              # coef > 0
        out <- x < (stats[2] - coef * iqr) | x > (stats[4] + coef * iqr)
        if(any(out[nna])) stats[c(1, 5)] <- range(x[!out], na.rm = TRUE)
    }
    conf <- NULL
    if(do.conf && is.null(weights))
        conf <- stats[3] + c(-1.58, 1.58) * iqr / sqrt(n)
    if(do.conf&& !is.null(weights))
        conf <- stats[3] + c(-1.58, 1.58) * iqr * sqrt(sum((weights/sum(weights))^2))
    nn<-ifelse(is.null(weights),n,sum(weights))
            list(stats = stats, n = nn, conf = conf,
                out = if(do.out) x[out & nna] else numeric(0))
}

wtd.fivenum.tmr <-
function(x, weights=NULL, na.rm=TRUE)
{
    interpolatedindex<-function(myval,weights){
        indices<-1:length(weights)
        n<-sum(weights)
        weightsbelow<-rep(0,length(weights))
        for (i in 2:length(weights))
            weightsbelow[i] <- weightsbelow[i-1]+weights[i-1]
        weightsabove<-n-weightsbelow-weights
        lowcands<-weightsbelow<myval
        highcands<-weightsabove<n-myval
        (ifelse(any(lowcands),max(indices[lowcands]),1)+
         ifelse(any(highcands),min(indices[highcands]),length(x)))/2
    }
    if (is.null(weights)) weights<-rep(1,length(x))
    if (length(x)>1)
        equalweights<- all((weights[2:length(weights)]-
              weights[1:length(weights)-1])==0)
    else
        equalweights<-TRUE
    xna <- (is.na(x) | weights==0)
    if(na.rm) x <- x[!xna]
    else if(any(xna)) return(rep.int(NA,5))
    sortorder<-order(x)
    x <- x[sortorder]
    weights<-weights[sortorder]
    n <- sum(weights)
    if(n == 0) rep.int(NA,5)
    else {
      if (equalweights){
          d <- c(1, 0.5*floor(0.5*(n+3)), 0.5*(n+1),
                  n+1-0.5*floor(0.5*(n+3)), n)
      }
      else {
        if(length(x)>1)
            d<-c(1,sapply(c(0.25*n,0.5*n,0.75*n),
                function(xxx)interpolatedindex(xxx,weights)),
                length(x))
        else
            d<-rep(1,5)
      }
      0.5*(x[floor(d)]+x[ceiling(d)])
    }
}




## Stuff from Hmisc
## included here to avoid issues with dependence on Hmisc

## See stackoverflow.com/questions/10049402

wtd.mean <- function(x, weights=NULL, normwt='ignored', na.rm=TRUE)
{
  if(! length(weights)) return(mean(x, na.rm=na.rm))
  if(na.rm) {
    s <- ! is.na(x + weights)
    x <- x[s]
    weights <- weights[s]
  }

  sum(weights * x) / sum(weights)
}



wtd.var <- function(x, weights=NULL, normwt=FALSE, na.rm=TRUE,
                    method = c('unbiased', 'ML'))
  ## By Benjamin Tyner <btyner@gmail.com> 2017-0-12
{
  method <- match.arg(method)
  if(! length(weights)) {
    if(na.rm) x <- x[!is.na(x)]
    return(var(x))
  }

  if(na.rm) {
    s       <- !is.na(x + weights)
    x       <- x[s]
    weights <- weights[s]
  }

  if(normwt)
    weights <- weights * length(x) / sum(weights)

  if(normwt || method == 'ML')
    return(as.numeric(stats::cov.wt(cbind(x), weights, method = method)$cov))

  # the remainder is for the special case of unbiased frequency weights
  sw  <- sum(weights)
  if(sw <= 1)
      warning("only one effective observation; variance estimate undefined")

  xbar <- sum(weights * x) / sw
  sum(weights*((x - xbar)^2)) / (sw - 1)
}



### wtd.quantile <- function(x, weights=NULL, probs=c(0, .25, .5, .75, 1),
###                          type=c('quantile','(i-1)/(n-1)','i/(n+1)','i/n'),
###                          normwt=FALSE, na.rm=TRUE)
### {
###   if(! length(weights))
###     return(quantile(x, probs=probs, na.rm=na.rm))
###
###   type <- match.arg(type)
###   if(any(probs < 0 | probs > 1))
###     stop("Probabilities must be between 0 and 1 inclusive")
###
###   nams <- paste(format(round(probs * 100, if(length(probs) > 1)
###                              2 - log10(diff(range(probs))) else 2)),
###                 "%", sep = "")
###
###   i <- is.na(weights) | weights == 0
###   if(any(i)) {
###     x <- x[! i]
###     weights <- weights[! i]
###     }
###   if(type == 'quantile') {
###     w <- wtd.table(x, weights, na.rm=na.rm, normwt=normwt, type='list')
###     x     <- w$x
###     wts   <- w$sum.of.weights
###     n     <- sum(wts)
###     order <- 1 + (n - 1) * probs
###     low   <- pmax(floor(order), 1)
###     high  <- pmin(low + 1, n)
###     order <- order %% 1
###     ## Find low and high order statistics
###     ## These are minimum values of x such that the cum. freqs >= c(low,high)
###     allq <- approx(cumsum(wts), x, xout=c(low,high),
###                    method='constant', f=1, rule=2)$y
###     k <- length(probs)
###     quantiles <- (1 - order)*allq[1:k] + order*allq[-(1:k)]
###     names(quantiles) <- nams
###     return(quantiles)
###   }
###   w <- wtd.Ecdf(x, weights, na.rm=na.rm, type=type, normwt=normwt)
###   structure(approx(w$ecdf, w$x, xout=probs, rule=2)$y,
###             names=nams)
### }
###
###
### wtd.Ecdf <- function(x, weights=NULL,
###                      type=c('i/n','(i-1)/(n-1)','i/(n+1)'),
###                      normwt=FALSE, na.rm=TRUE)
### {
###   type <- match.arg(type)
###   switch(type,
###          '(i-1)/(n-1)'={a <- b <- -1},
###          'i/(n+1)'    ={a <- 0; b <- 1},
###          'i/n'        ={a <- b <- 0})
###
###   if(! length(weights)) {
###     ##.Options$digits <- 7  ## to get good resolution for names(table(x))
###     oldopt <- options('digits')
###     options(digits=7)
###     on.exit(options(oldopt))
###     cumu <- table(x)    ## R does not give names for cumsum
###     isdate <- testDateTime(x)  ## 31aug02
###     ax <- attributes(x)
###     ax$names <- NULL
###     x <- as.numeric(names(cumu))
###     if(isdate) attributes(x) <- c(attributes(x),ax)
###     cumu <- cumsum(cumu)
###     cdf <- (cumu + a)/(cumu[length(cumu)] + b)
###     if(cdf[1]>0) {
###       x <- c(x[1], x);
###       cdf <- c(0,cdf)
###     }
###
###     return(list(x = x, ecdf=cdf))
###   }
###
###   w <- wtd.table(x, weights, normwt=normwt, na.rm=na.rm)
###   cumu <- cumsum(w$sum.of.weights)
###   cdf <- (cumu + a)/(cumu[length(cumu)] + b)
###   list(x = c(if(cdf[1]>0) w$x[1], w$x), ecdf=c(if(cdf[1]>0)0, cdf))
### }


### wtd.table <- function(x, weights=NULL, type=c('list','table'),
###                       normwt=FALSE, na.rm=TRUE)
### {
###   type <- match.arg(type)
###   if(! length(weights))
###     weights <- rep(1, length(x))
###
###   isdate <- testDateTime(x)  ## 31aug02 + next 2
###   ax <- attributes(x)
###   ax$names <- NULL
###
###   if(is.character(x)) x <- as.factor(x)
###   lev <- levels(x)
###   x <- unclass(x)
###
###   if(na.rm) {
###     s <- ! is.na(x + weights)
###     x <- x[s, drop=FALSE]    ## drop is for factor class
###     weights <- weights[s]
###   }
###
###   n <- length(x)
###   if(normwt)
###     weights <- weights * length(x) / sum(weights)
###
###   i <- order(x)  # R does not preserve levels here
###   x <- x[i]; weights <- weights[i]
###
###   if(anyDuplicated(x)) {  ## diff(x) == 0 faster but doesn't handle Inf  '
###     weights <- tapply(weights, x, sum)
###     if(length(lev)) {
###       levused <- lev[sort(unique(x))]
###       if((length(weights) > length(levused)) &&
###          any(is.na(weights)))
###         weights <- weights[! is.na(weights)]
###
###       if(length(weights) != length(levused))
###         stop('program logic error')
###
###       names(weights) <- levused
###     }
###
###     if(! length(names(weights)))
###       stop('program logic error')
###
###     if(type=='table')
###       return(weights)
###
###     x <- all.is.numeric(names(weights), 'vector')
###     if(isdate)
###       attributes(x) <- c(attributes(x),ax)
###
###     names(weights) <- NULL
###     return(list(x=x, sum.of.weights=weights))
###   }
###
###   xx <- x
###   if(isdate)
###     attributes(xx) <- c(attributes(xx),ax)
###
###   if(type=='list')
###     list(x=if(length(lev))lev[x]
###            else xx,
###          sum.of.weights=weights)
###   else {
###     names(weights) <- if(length(lev)) lev[x]
###                       else xx
###     weights
###   }
### }


### wtd.rank <- function(x, weights=NULL, normwt=FALSE, na.rm=TRUE)
### {
###   if(! length(weights))
###     return(rank(x, na.last=if(na.rm) NA else TRUE))
###
###   tab <- wtd.table(x, weights, normwt=normwt, na.rm=na.rm)
###
###   freqs <- tab$sum.of.weights
###   ## rank of x = # <= x - .5 (# = x, minus 1)
###   r <- cumsum(freqs) - .5*(freqs-1)
###   ## Now r gives ranks for all unique x values.  Do table look-up
###   ## to spread these ranks around for all x values.  r is in order of x
###   approx(tab$x, r, xout=x)$y
### }
###
###
### wtd.loess.noiter <- function(x, y, weights=rep(1,n),
###                              span=2/3, degree=1, cell=.13333,
###                              type=c('all','ordered all','evaluate'),
###                              evaluation=100, na.rm=TRUE) {
###   type <- match.arg(type)
###   n <- length(y)
###   if(na.rm) {
###     s <- ! is.na(x + y + weights)
###     x <- x[s]; y <- y[s]; weights <- weights[s]; n <- length(y)
###   }
###
###   max.kd <- max(200, n)
###   # y <- stats:::simpleLoess(y, x, weights=weights, span=span,
###   #                          degree=degree, cell=cell)$fitted
###   y <- fitted(loess(y ~ x, weights=weights, span=span, degree=degree,
### 		control=loess.control(cell=cell, iterations=1)))
###
###   switch(type,
###          all=list(x=x, y=y),
###          'ordered all'={
###            i <- order(x);
###            list(x=x[i],y=y[i])
###          },
###          evaluate={
###            r <- range(x, na.rm=na.rm)
###            approx(x, y, xout=seq(r[1], r[2], length=evaluation))
###          })
### }
###
### num.denom.setup <- function(num, denom)
### {
###   n <- length(num)
###   if(length(denom) != n)
###     stop('lengths of num and denom must match')
###
###   s <- (1:n)[! is.na(num + denom) & denom != 0]
###   num <- num[s];
###   denom <- denom[s]
###
###   subs <- s[num > 0]
###   y <- rep(1, length(subs))
###   wt <- num[num > 0]
###   other <- denom - num
###   subs <- c(subs, s[other > 0])
###   wt <- c(wt, other[other > 0])
###   y <- c(y, rep(0, sum(other>0)))
###   list(subs=subs, weights=wt, y=y)
### }


### ## Determine if variable is a date, time, or date/time variable in R.
### ## The following 2 functions are used by describe.vector
### ## timeUsed assumes is date/time combination variable and has no NAs
### testDateTime <- function(x, what=c('either','both','timeVaries'))
### {
###   what <- match.arg(what)
###   cl <- class(x)
###   if(!length(cl))
###     return(FALSE)
###
###   dc <- c('Date', 'POSIXt','POSIXct','dates','times','chron')
###
###   dtc <- c('POSIXt','POSIXct','chron')
###
###   switch(what,
###          either = any(cl %in% dc),
###          both   = any(cl %in% dtc),
###          timeVaries = {
###            if('chron' %in% cl || 'Date' %in% cl) {
###              ## chron or S+ timeDate
###              y <- as.numeric(x)
###              length(unique(round(y - floor(y),13))) > 1
###            }
###            else length(unique(format(x,'%H%M%S'))) > 1
###          })
### }
###
### ###########################
### all.is.numeric <- function(x, what=c('test','vector'),
###                            extras=c('.','NA'))
### {
###   what <- match.arg(what)
###   x <- sub('[[:space:]]+$', '', x)
###   x <- sub('^[[:space:]]+', '', x)
###   xs <- x[x %nin% c('',extras)]
###   if(! length(xs)) return(if(what == 'test') FALSE else x)
###   isnum <- suppressWarnings(!any(is.na(as.numeric(xs))))
###   if(what=='test')
###     isnum
###   else if(isnum)
###     as.numeric(x)
###   else x
### }
