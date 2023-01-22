import React from "react";
import {useEffect, useState} from "react";
import ReactLoading from "react-loading"
import axios from "axios";



function Home() {
  const [getSubmit, setSubmit] = React.useState(false)
  const [getLoading, setLoading] = React.useState(false)
  const [getResult, setResult] = React.useState([])
  const [getImage, setImage] = React.useState([])
  const [getImageToSend, setImageToSend] = React.useState([])
  const [getImageURL, setImageURL] = React.useState([])
  const [getPer_1, setPer_1] = React.useState([])
  const [getPer_2, setPer_2] = React.useState([])

  useEffect(() => {
    setSubmit(false)
    setImageToSend([])
    setImage([])
    setResult([])
  }, [])

  useEffect(() => {
    if (getImage.length != 0) {
      console.log(typeof getImage[0])
      const imageURL = [URL.createObjectURL(getImage[0])]
      setImageURL(imageURL)
      setImageToSend(getImage)
      setImage([])
    }
  }, [getImage])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (getImageURL.length != 0) {
      setSubmit(true)
      setLoading(true)
      setImageURL([])
      sendImage()
    }    
  }

  const handleReset = (e) => {
    e.preventDefault()
    setSubmit(false)
    setImageToSend([])
    setImage([])
    setResult([])
  }

  async function sendImage() {
    console.log(getImageToSend[0])
    let formData = new FormData();
    formData.append("image", getImageToSend[0]);
    axios.post("/send_image", formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    .then((response) => {
      setLoading(false)
      if (response.status == '200') {
        console.log(response)
        setResult([response.data['output']])
        setPer_1([response.data['per_1']])
        setPer_2([response.data['per_2']])
      }

    });
    // fetch('/send_image', {
    //   method: 'POST',
    //   body: formData,
    // }).then((response) => {
    //   setLoading(false)
    //   if (response.status == '200') {
    //     console.log(response.text())
    //     setResult([response.data['output']])
    //   }
    // });
  }


    return (
      <div className="App">
        <header className="App-body">
        {(getResult.length != 0) && 
          <div>
            <p className="h4 text-center p-5">Returned from Network!</p>
            {(getResult[0] == 0) &&
              <div>
                <p className="h4 text-center">Image does show Terry Crews!</p>
                <p className="h4 text-center">{"Image is "+getPer_1+"% Terry Crews"}</p>
              </div>
            }
            {(getResult[0] == 1) &&
              <div>
                <p className="h4 text-center">Image does show not Terry Crews!</p>
                <p className="h4 text-center">{"Image is "+getPer_2+"% not Terry Crews"}</p>
              </div>
            }
            <div className="p-5">
              <button className="btn btn-primary" onClick={handleReset}>Submit Another Image</button>
            </div>
          </div>
        }
          {getLoading &&
           <div className="h-25 w-100 p-5 row align-items-center justify-content-center">
            <ReactLoading
              type={"spinningBubbles"}
              color={"#D3D3D3"}
              height={100}
              width={100}
            />
            <p className="h4 text-center p-5">Running the Neural Network...</p>
            </div>
          }
          {!getSubmit && 
          <div>
            <p className="h4 text-center p-5">Upload your Image!</p>
          </div>
          }
          {!getSubmit && 
          <div className="row align-items-center justify-content-center">
            
              <div className="form-group">
                <form onSubmit={handleSubmit}>
                  <label className="label">
                    <input
                      type="file"
                      accept="image/*"
                      className="form-control input"
                      onChange={(e) => setImage(e.target.files)}

                    />
                  
                  </label>
                  <label className="label">
                    <button type="submit" className="btn btn-primary">Submit</button>
                  </label>
                </form>
              </div>
          </div>
          }
          {(getImageURL.length != 0) &&
            <img className="img-responsive img-thumbnail" src={getImageURL}/>

          }

        </header>
      </div>
    );
  }


  export default Home;