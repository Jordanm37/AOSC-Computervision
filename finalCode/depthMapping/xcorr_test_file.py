from scipy import misc
from matplotlib.pyplot import subplots

from test_xcorr import TemplateMatch


image = misc.face()
template = image[240:281,240:281]
TM = TemplateMatch(template,method='both')
ncc,ssd = TM(image)
nccloc = np.nonzero(ncc == ncc.max())
ssdloc = np.nonzero(ssd == ssd.min())

fig,[[ax1,ax2],[ax3,ax4]] = subplots(2,2,num='ND Template Search')
ax1.imshow(image,interpolation='nearest')
ax1.set_title('Search image')
ax2.imshow(template,interpolation='nearest')
ax2.set_title('Template')
ax3.hold(True)
ax3.imshow(ncc,interpolation='nearest')
ax3.plot(nccloc[1],nccloc[0],'w+')
ax3.set_title('Normalized cross-correlation')
ax4.hold(True)
ax4.imshow(ssd,interpolation='nearest')
ax4.plot(ssdloc[1],ssdloc[0],'w+')
ax4.set_title('Sum of squared differences')
fig.tight_layout()
fig.canvas.draw()



