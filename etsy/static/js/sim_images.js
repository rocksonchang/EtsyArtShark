top_id = {{top_id | tojson}}
top_sims = {{top_sims | tojson}}

for(var i=0 ; i<3; i=i+1)
{
    document.write('<p>');
    for(var j=0 ; j<4; j=j+1)
    {
        document.write('<div class="floated_img">')            
        document.write('<img src="/static/assets/pa_med_1/' + top_id[i+3*j] + '" alt="Some image" style="float:left">')          
        document.write('</div>')
    }
    document.write('</p>');
}        
document.write('</br>') 
document.write('<p>' + top_sims.slice(0,12) + '</p>')