require(ggplot2)
require(ggrepel)
require(tikzDevice)

coords <- data.frame(read.csv('mapcoords.csv',header=F,sep=' '))
colnames(coords) <- c('lang','lat','lon','pdir','pln')

#inner-outer ground truth
pgold <- c(1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,0,0,0,0,1,0)

coords$pgold <- pgold


tikz('tex/component_map_groundtruth.tex')
ggplot()+
    borders(colour='lightgray',fill='lightgray')+coord_cartesian(xlim=c(-1.5,1.5)+range(coords$lon),ylim=c(-1.5,1.5)+range(coords$lat)) +
    geom_point(data=coords,aes(x=lon,y=lat,color=pgold)) + scale_color_gradient(limits=c(0,1)) + labs(color='$P(k=1)$') +
    geom_text_repel(data=coords,aes(x=lon,y=lat,label=lang),cex=2)+theme_bw()

dev.off()


tikz('tex/component_map_dir.tex')
ggplot()+
    borders(colour='lightgray',fill='lightgray')+coord_cartesian(xlim=c(-1.5,1.5)+range(coords$lon),ylim=c(-1.5,1.5)+range(coords$lat)) +
    geom_point(data=coords,aes(x=lon,y=lat,color=pdir)) + 
	#scale_color_gradient(limits=c(0,1)) + 
	labs(color='$P(k=1)$') +
    geom_text_repel(data=coords,aes(x=lon,y=lat,label=lang),cex=2)+theme_bw()

dev.off()


tikz('tex/component_map_ln.tex')
ggplot()+
    borders(colour='lightgray',fill='lightgray')+coord_cartesian(xlim=c(-1.5,1.5)+range(coords$lon),ylim=c(-1.5,1.5)+range(coords$lat)) +
    geom_point(data=coords,aes(x=lon,y=lat,color=pln)) + 
	#scale_color_gradient(limits=c(0,1)) + 
	labs(color='$P(k=1)$') +
    geom_text_repel(data=coords,aes(x=lon,y=lat,label=lang),cex=2)+theme_bw()

dev.off()
