import React from "react";
import { Link, withRouter } from "react-router-dom";

function Navigation(props) {
    return (
        <div className="navigation">
            <nav className = "navbar navbar-expand navbar-dark bg-dark">
                <div className="container">
                    <Link className = "navbar-brand" to="/">
                        isitterrycrews.com
                    </Link>
                    <div>
                        <ul className="navbar-nav ml-auto">
                            <li>
                                <Link className = "nav-link" to ='/'>
                                    Submit an Image
                                </Link>
                            </li>
                            <li>
                                <Link className = "nav-link" to ='/about'>
                                    About
                                </Link>
                            </li>
                        </ul>
                    </div>
                </div>
            </nav>
        </div>
    );
}

export default Navigation;