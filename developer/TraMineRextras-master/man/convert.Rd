\encoding{latin1}
\name{convert}
\alias{convert.g}

\title{Converting between graphical formats}


\description{
Wrapper function for converting graphics with ImageMagick}

\usage{
convert.g(path = NULL, fileroot= "*", from = "pdf",
    to = "png", create.path = TRUE, options = NULL)
}

\arguments{
  \item{path}{String: Path to the from graphic files. If \code{NULL} (default), the current path is used.}

  \item{fileroot}{String: Graphic root name; default is "*" for all files with the \code{from} extension.}

  \item{from}{String: File type extension specifying the from format.}

  \item{to}{String: File type extension specifying the to format.}

  \item{create.path}{Logical: Should the output files be placed in a \code{to} subfolder.}

  \item{options}{Additional options to be passed to the ImageMagick \code{mogrify} function}

}

\details{
    Conversion is done through a call to ImageMagick \code{mogrify} function. This means that ImageMagick should be installed on your system. It must also be listed in the path.

    For values such as \code{"pdf"} and \code{"eps"} of the \code{from} or \code{to} arguments ImageMagick works in conjunction with Gostscript. The latter should, therefore, also be accessible.
    
    ImageMagick is not suited for vector to vector format conversion because it rasterizes the image before the conversion. Therefore, \code{convert.g} is not suited too for such conversions.
}

\seealso{
\code{\link{png}}, \code{\link{pdf}}
}

\examples{
\dontrun{
## Convert all .pdf graphics in the "figSW" directory
## into .png files and put the files in a "png" subfolder.
convert.g(path="figSW", from="pdf", to="png")

## Same, but convert to .jpg files.
convert.g(path="figSW", to="jpg")

## convert file "example.jpg" in current path to ".pdf"
## and put it in same folder.
convert.g(fileroot = "example", from = "jpg", create.path=FALSE)
}


}
\keyword{util}
