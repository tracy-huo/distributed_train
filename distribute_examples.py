1#coding =utf-8

import numpy as np

import tensorflow as tf

#Define parameters
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate",0.00003,"Initial learning rate.")
tf.app.flags.DEFINE_integer("step_to_validate",1000,"Steps to step_to_validate and print loss")


#for distributed

tf.app.flags.DEFINE_string("ps_hosts" ,"","Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("ps_hosts", "","Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name" , "","One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index" , 0,"Index of task within the job")
tf.app.flags.DEFINE_integer("issync" , 0," 是否采用分布式的同步模式，1表示同步模式，0表示异步模式")


#Hyperparameters
learning_rate = FLAGS.learning_rate
step_to_validate = FLAGS.step_to_validate


def main(_):
    ps_hosts  = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps":ps_hosts , "worker":worker_hosts})
    server = tf.train.Sever(cluster, job_name = FLAGS.job_name,task_index=FLAGS.task_index)


    issync = FLAGS.issync
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name =="worker":
        with tf.device(tf.train.replica_device_setter(
                        worker_divice = "/job:worker/task:%d" % FLAGS.task_index,
                        cluster = cluster

        )):
        global_step = tf.Variable(0,name = " global_step",trainabel = False)

        input  = tf.placeholder("float")
        label  = tf.placeholder("flaot“)

        weight = tf.get_variable("weight",[1],tf.float32, initializer = tf.random_normal_initializer())
        bias = tf.get_variable("bias",[1],tf.float32, initializer = tf.random_normal_initializer())

        pred = tf.multiply(input,weight)+bias


        loss_value = loss(label,pred)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        grads_and_vars = optimizer.compute_gradients(loss_value)
        if issync ==1:
            rep_op = tf.train.SyncReplicasOptimizer(optimizer,
                                                    replicas_to_agregate = len(
                                                        worker_hosts),
                                                        replica_id = FLAGS.task_index,
                                                        total_num_replicas = len(
                                                            worker_hosts),
                                                            use_locking = True)
            train_op = rep_op.apply_gradients(grads_and_vars，global_step = global_step)
            init_token_op = rep_op.get_init_tokens_op()
            chief_queue_runner = rep.op.get_chief_queue_runner()
        else:
            #以异步模式计算更新梯度
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                global_step = global_step)


            init_op = tf.initialize_all_variables()

            saver = tf.train.Saver()
            tf.summary.scalar("cost" ,loss_value)
            summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(is_chief = (FLAGS.task_index == 0),
                                logdir = "./checkpoint/",
                                init_op = init_op,
                                summary_op = None,
                                saver = saver,
                                global_step = global_step,
                                save_model_secs = 60
        )
        with sv.prepare_or_wait_for_session(server.target) as sess:
            # 如果是同步模式
            if FLAGS.task_index == 0 and issync == 1:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)
            step = 0
             while  step < 1000000:
                 train_x = np.random.randn(1)
                 train_y = 2 * train_x + np.random.randn(1) * 0.33  + 10
                 _, loss_v, step = sess.run([train_op, loss_value,global_step], feed_dict={input:train_x, label:train_y})
                if step % steps_to_validate == 0:
                    w,b = sess.run([weight,biase])
                    print("step: %d, weight: %f, biase: %f, loss: %f" %(step, w, b, loss_v))

        sv.stop()


def loss(label.pred):
    reuturn tf.square(label - pred)


if __name__ == "__main__":
    tf.app.run()


