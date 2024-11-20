import React, { useState } from 'react';
import axios from 'axios';

function VideoTranscriber() {
    const [videoUrl, setVideoUrl] = useState("");
    const [question, setQuestion] = useState("");
    const [answer, setAnswer] = useState("");
    const [loading, setLoading] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalMessage, setModalMessage] = useState("");
    const [answerHistory, setAnswerHistory] = useState([]);

    const handleTranscribe = async () => {
        setLoading(true);
        try {
            await axios.post("http://localhost:5000/transcribe", { videoUrl });
            setModalMessage("Transcription started! You can ask questions now.");
            setIsModalOpen(true);
            // Clear answer history when new video is transcribed
            setAnswerHistory([]);
        } catch (error) {
            console.error("Error transcribing video:", error);
            setModalMessage("Failed to transcribe the video. Please try again.");
            setIsModalOpen(true);
        } finally {
            setLoading(false);
        }
    };

    const handleQuery = async () => {
        setLoading(true);
        try {
            const response = await axios.post("http://localhost:5000/query", { question });
            const newAnswer = response.data.response;
            setAnswer(newAnswer);
            
            // Add new question and answer to history
            setAnswerHistory(prev => [...prev, {
                question,
                answer: newAnswer,
                timestamp: new Date().toLocaleTimeString()
            }]);
            
            setQuestion(""); // Clear question input after successful query
            console.log("Answer:", newAnswer);
        } catch (error) {
            console.error("Error querying answer:", error);
            setModalMessage("Failed to retrieve the answer. Please try again.");
            setIsModalOpen(true);
        } finally {
            setLoading(false);
        }
    };

    const closeModal = () => {
        setIsModalOpen(false);
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-gray-200 to-gray-400 p-4">
            <div className="max-w-3xl w-full bg-white shadow-2xl rounded-lg p-8 space-y-6">
                <h2 className="text-3xl font-extrabold text-gray-800 text-center mb-6">
                    Video Transcription & QA
                </h2>

                {/* Video URL Input */}
                <div className="space-y-2">
                    <label className="text-lg font-semibold text-gray-700">YouTube Video Link</label>
                    <input
                        type="text"
                        value={videoUrl}
                        onChange={(e) => setVideoUrl(e.target.value)}
                        placeholder="Enter YouTube video link"
                        className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm transition duration-300 ease-in-out"
                    />
                </div>
                <button
                    onClick={handleTranscribe}
                    disabled={!videoUrl || loading}
                    className="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 transition disabled:opacity-50"
                >
                    {loading ? (
                        <div className="flex justify-center items-center">
                            <svg className="animate-spin h-5 w-5 mr-3 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeDasharray="150" strokeDashoffset="0" />
                            </svg>
                            Processing...
                        </div>
                    ) : (
                        "Transcribe Video"
                    )}
                </button>

                {/* Question Input */}
                <div className="space-y-2">
                    <label className="text-lg font-semibold text-gray-700">Ask a Question</label>
                    <input
                        type="text"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        placeholder="Ask a question about the video"
                        className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 shadow-sm transition duration-300 ease-in-out"
                    />
                </div>
                <button
                    onClick={handleQuery}
                    disabled={!question || loading}
                    className="w-full bg-green-600 text-white font-semibold py-3 px-6 rounded-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 transition disabled:opacity-50"
                >
                    {loading ? (
                        <div className="flex justify-center items-center">
                            <svg className="animate-spin h-5 w-5 mr-3 text-white" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">
                                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" strokeDasharray="150" strokeDashoffset="0" />
                            </svg>
                            Getting Answer...
                        </div>
                    ) : (
                        "Get Answer"
                    )}
                </button>

                {/* Answer History Section */}
                {answerHistory.length > 0 && (
                    <div className="mt-8">
                        <h3 className="text-2xl font-bold text-gray-800 mb-4">Answer History</h3>
                        <div className="space-y-4">
                            {answerHistory.map((item, index) => (
                                <div key={index} className="bg-gray-50 rounded-lg p-4 shadow-md">
                                    <div className="flex justify-between items-start mb-2">
                                        <p className="font-semibold text-gray-700">Q: {item.question}</p>
                                        <span className="text-sm text-gray-500">{item.timestamp}</span>
                                    </div>
                                    <p className="text-gray-600 mt-2">A: {item.answer}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Modal Pop-up */}
                {isModalOpen && (
                    <div className="fixed inset-0 bg-gray-800 bg-opacity-50 flex justify-center items-center z-50">
                        <div className="bg-white p-8 rounded-lg shadow-lg max-w-sm w-full">
                            <h3 className="text-lg font-semibold text-gray-800">{modalMessage}</h3>
                            <div className="mt-4 flex justify-end">
                                <button
                                    onClick={closeModal}
                                    className="bg-blue-500 text-white font-semibold py-2 px-4 rounded-md hover:bg-blue-600 transition"
                                >
                                    Close
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default VideoTranscriber;